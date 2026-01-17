"""
Entity Resolution Matching Service.
Combines deterministic (exact/fuzzy) and probabilistic (Splink) matching.
"""

import logging
import time
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import asyncio
from dataclasses import dataclass

import pandas as pd
import duckdb
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.schemas import (
    RecordInput, MatchResult, MatchType, PersonRecord, 
    FieldScore, MatchResponse
)

logger = logging.getLogger(__name__)
settings = get_settings()

# Thread pool for CPU-bound Splink operations
executor = ThreadPoolExecutor(max_workers=4)


@dataclass
class MatchConfig:
    """Configuration for a matching operation."""
    match_type: MatchType
    threshold: float
    max_results: int
    include_scores: bool


class EntityMatcher:
    """
    Main entity resolution service.
    Handles blocking, candidate generation, and scoring.
    """
    
    def __init__(self):
        self.splink_linker = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the Splink linker with trained model."""
        if self._initialized:
            return
        
        # Splink initialization happens lazily on first match
        self._initialized = True
        logger.info("EntityMatcher initialized")
    
    async def match_single(
        self,
        session: AsyncSession,
        record: RecordInput,
        config: MatchConfig,
    ) -> MatchResponse:
        """
        Match a single record against the database.
        
        Pipeline:
        1. Generate blocking keys from input
        2. Retrieve candidates via SQL (deterministic blocking)
        3. Score candidates (fuzzy + probabilistic)
        4. Return ranked results
        """
        start_time = time.perf_counter()
        
        # Step 1 & 2: Get candidates using blocking
        candidates = await self._get_candidates(session, record, config)
        
        if not candidates:
            return MatchResponse(
                query=record,
                matches=[],
                total_matches=0,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                match_type=config.match_type,
                threshold_used=config.threshold,
            )
        
        # Step 3: Score candidates
        if config.match_type == MatchType.DETERMINISTIC_EXACT:
            scored = self._score_deterministic_exact(record, candidates)
        elif config.match_type == MatchType.DETERMINISTIC_FUZZY:
            scored = self._score_deterministic_fuzzy(record, candidates)
        elif config.match_type == MatchType.PROBABILISTIC:
            scored = await self._score_probabilistic(record, candidates)
        else:  # HYBRID
            scored = await self._score_hybrid(record, candidates)
        
        # Step 4: Filter and sort by threshold
        matches = [
            m for m in scored 
            if m.overall_score >= config.threshold
        ]
        matches.sort(key=lambda x: x.overall_score, reverse=True)
        matches = matches[:config.max_results]
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return MatchResponse(
            query=record,
            matches=matches,
            total_matches=len(matches),
            processing_time_ms=processing_time,
            match_type=config.match_type,
            threshold_used=config.threshold,
        )
    
    async def _get_candidates(
        self,
        session: AsyncSession,
        record: RecordInput,
        config: MatchConfig,
    ) -> List[dict]:
        """
        Retrieve candidate records using blocking strategies.
        Uses multiple blocking rules for high recall.
        """
        # Build blocking conditions
        blocking_conditions = []
        params = {}
        
        # Block 1: Exact SSN last 4 (if provided)
        if record.ssn_last4:
            blocking_conditions.append("ssn_last4 = :ssn_last4")
            params["ssn_last4"] = record.ssn_last4
        
        # Block 2: Soundex of last name + birth year
        if record.last_name and record.birth_date:
            blocking_conditions.append(
                "(soundex(last_name) = soundex(:last_name) AND "
                "EXTRACT(YEAR FROM birth_date) = :birth_year)"
            )
            params["last_name"] = record.last_name
            params["birth_year"] = record.birth_date.year
        
        # Block 3: Trigram similarity on last name (fuzzy)
        if record.last_name:
            blocking_conditions.append(
                "last_name % :last_name_trgm"
            )
            params["last_name_trgm"] = record.last_name
        
        # Block 4: First name + State
        if record.first_name and record.state:
            blocking_conditions.append(
                "(soundex(first_name) = soundex(:first_name) AND state = :state)"
            )
            params["first_name"] = record.first_name
            params["state"] = record.state.upper()
        
        if not blocking_conditions:
            # Fallback: just use trigram on name
            blocking_conditions.append("last_name % :last_name_fallback")
            params["last_name_fallback"] = record.last_name
        
        # Combine with OR for maximum recall
        where_clause = " OR ".join(f"({c})" for c in blocking_conditions)
        
        query = text(f"""
            SELECT 
                id, first_name, middle_name, last_name,
                birth_date, death_date, ssn_last4, city, state,
                similarity(last_name, :sim_last_name) as name_sim
            FROM persons
            WHERE {where_clause}
            ORDER BY name_sim DESC
            LIMIT :limit
        """)
        
        params["sim_last_name"] = record.last_name
        params["limit"] = config.max_results * 3  # Get more candidates for scoring
        
        result = await session.execute(query, params)
        rows = result.fetchall()
        
        candidates = []
        for row in rows:
            candidates.append({
                "id": str(row.id),
                "first_name": row.first_name,
                "middle_name": row.middle_name,
                "last_name": row.last_name,
                "birth_date": row.birth_date.isoformat() if row.birth_date else None,
                "death_date": row.death_date.isoformat() if row.death_date else None,
                "ssn_last4": row.ssn_last4,
                "city": row.city,
                "state": row.state,
                "name_sim": float(row.name_sim) if row.name_sim else 0.0,
            })
        
        logger.debug(f"Retrieved {len(candidates)} candidates for matching")
        return candidates
    
    def _score_deterministic_exact(
        self,
        record: RecordInput,
        candidates: List[dict],
    ) -> List[MatchResult]:
        """Score using exact field matching only."""
        results = []
        
        for cand in candidates:
            field_scores = []
            total_score = 0.0
            total_weight = 0.0
            
            # Define field weights
            weights = {
                "ssn_last4": 0.30,
                "last_name": 0.20,
                "first_name": 0.15,
                "birth_date": 0.20,
                "city": 0.08,
                "state": 0.07,
            }
            
            for field, weight in weights.items():
                input_val = getattr(record, field, None)
                cand_val = cand.get(field)
                
                if input_val and cand_val:
                    # Handle date comparison
                    if field == "birth_date":
                        input_str = input_val.isoformat() if hasattr(input_val, 'isoformat') else str(input_val)
                        cand_str = str(cand_val)
                        score = 1.0 if input_str == cand_str else 0.0
                    else:
                        # String comparison (case-insensitive)
                        score = 1.0 if str(input_val).lower() == str(cand_val).lower() else 0.0
                    
                    field_scores.append(FieldScore(
                        field=field,
                        score=score,
                        match_level="exact" if score == 1.0 else "no_match",
                        weight=weight,
                    ))
                    total_score += score * weight
                    total_weight += weight
            
            overall = total_score / total_weight if total_weight > 0 else 0.0
            
            results.append(MatchResult(
                record=PersonRecord(
                    id=cand["id"],
                    first_name=cand["first_name"],
                    middle_name=cand.get("middle_name"),
                    last_name=cand["last_name"],
                    birth_date=cand.get("birth_date"),
                    death_date=cand.get("death_date"),
                    ssn_last4=cand.get("ssn_last4"),
                    city=cand.get("city"),
                    state=cand.get("state"),
                ),
                overall_score=overall,
                match_probability=None,
                field_scores=field_scores,
                match_type_used=MatchType.DETERMINISTIC_EXACT,
            ))
        
        return results
    
    def _score_deterministic_fuzzy(
        self,
        record: RecordInput,
        candidates: List[dict],
    ) -> List[MatchResult]:
        """Score using fuzzy field matching."""
        results = []
        
        for cand in candidates:
            field_scores = []
            total_score = 0.0
            total_weight = 0.0
            
            weights = {
                "ssn_last4": 0.30,
                "last_name": 0.20,
                "first_name": 0.15,
                "birth_date": 0.20,
                "city": 0.08,
                "state": 0.07,
            }
            
            for field, weight in weights.items():
                input_val = getattr(record, field, None)
                cand_val = cand.get(field)
                
                if input_val and cand_val:
                    score, level = self._fuzzy_compare(field, input_val, cand_val)
                    
                    field_scores.append(FieldScore(
                        field=field,
                        score=score,
                        match_level=level,
                        weight=weight,
                    ))
                    total_score += score * weight
                    total_weight += weight
            
            overall = total_score / total_weight if total_weight > 0 else 0.0
            
            results.append(MatchResult(
                record=PersonRecord(
                    id=cand["id"],
                    first_name=cand["first_name"],
                    middle_name=cand.get("middle_name"),
                    last_name=cand["last_name"],
                    birth_date=cand.get("birth_date"),
                    death_date=cand.get("death_date"),
                    ssn_last4=cand.get("ssn_last4"),
                    city=cand.get("city"),
                    state=cand.get("state"),
                ),
                overall_score=overall,
                match_probability=None,
                field_scores=field_scores,
                match_type_used=MatchType.DETERMINISTIC_FUZZY,
            ))
        
        return results
    
    def _fuzzy_compare(self, field: str, val1, val2) -> Tuple[float, str]:
        """Compare two values with fuzzy matching."""
        from difflib import SequenceMatcher
        
        # Handle dates
        if field == "birth_date":
            str1 = val1.isoformat() if hasattr(val1, 'isoformat') else str(val1)
            str2 = str(val2)
            if str1 == str2:
                return 1.0, "exact"
            # Check year match
            if str1[:4] == str2[:4]:
                return 0.7, "partial"
            return 0.0, "no_match"
        
        # Handle strings
        str1 = str(val1).lower().strip()
        str2 = str(val2).lower().strip()
        
        if str1 == str2:
            return 1.0, "exact"
        
        # Calculate similarity
        ratio = SequenceMatcher(None, str1, str2).ratio()
        
        if ratio >= 0.9:
            return ratio, "fuzzy"
        elif ratio >= 0.7:
            return ratio, "partial"
        else:
            return ratio, "no_match"
    
    async def _score_probabilistic(
        self,
        record: RecordInput,
        candidates: List[dict],
    ) -> List[MatchResult]:
        """Score using Splink probabilistic matching."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            self._run_splink_scoring,
            record,
            candidates,
        )
    
    def _run_splink_scoring(
        self,
        record: RecordInput,
        candidates: List[dict],
    ) -> List[MatchResult]:
        """Run Splink scoring in a thread (CPU-bound)."""
        try:
            from splink import Linker, SettingsCreator, block_on
            import splink.comparison_library as cl
            
            # Convert input record to dataframe
            input_df = pd.DataFrame([{
                "unique_id": "input_record",
                "first_name": record.first_name,
                "middle_name": record.middle_name,
                "last_name": record.last_name,
                "birth_date": record.birth_date.isoformat() if record.birth_date else None,
                "ssn_last4": record.ssn_last4,
                "city": record.city,
                "state": record.state,
            }])
            
            # Convert candidates to dataframe
            cand_df = pd.DataFrame([{
                "unique_id": c["id"],
                "first_name": c["first_name"],
                "middle_name": c.get("middle_name"),
                "last_name": c["last_name"],
                "birth_date": c.get("birth_date"),
                "ssn_last4": c.get("ssn_last4"),
                "city": c.get("city"),
                "state": c.get("state"),
            } for c in candidates])
            
            if cand_df.empty:
                return []
            
            # Configure Splink settings
            splink_settings = SettingsCreator(
                link_type="link_only",
                comparisons=[
                    cl.JaroWinklerAtThresholds("first_name", [0.9, 0.7]),
                    cl.JaroWinklerAtThresholds("last_name", [0.9, 0.7]),
                    cl.ExactMatch("ssn_last4"),
                    cl.DateOfBirthComparison("birth_date", input_is_string=True),
                    cl.ExactMatch("state"),
                    cl.JaroWinklerAtThresholds("city", [0.9]),
                ],
                blocking_rules_to_generate_predictions=[
                    block_on("ssn_last4"),
                    block_on("last_name"),
                ],
            )
            
            # Create linker with DuckDB backend
            db = duckdb.connect()
            linker = Linker(
                [input_df, cand_df],
                splink_settings,
                db_api=db,
            )
            
            # Get predictions
            predictions = linker.inference.predict()
            results_df = predictions.as_pandas_dataframe()
            
            # Convert to MatchResult objects
            results = []
            for _, row in results_df.iterrows():
                # Find the candidate record
                cand_id = row.get("unique_id_r") or row.get("unique_id_l")
                if cand_id == "input_record":
                    cand_id = row.get("unique_id_r")
                
                cand = next((c for c in candidates if c["id"] == cand_id), None)
                if not cand:
                    continue
                
                prob = row.get("match_probability", row.get("match_weight", 0.5))
                
                results.append(MatchResult(
                    record=PersonRecord(
                        id=cand["id"],
                        first_name=cand["first_name"],
                        middle_name=cand.get("middle_name"),
                        last_name=cand["last_name"],
                        birth_date=cand.get("birth_date"),
                        death_date=cand.get("death_date"),
                        ssn_last4=cand.get("ssn_last4"),
                        city=cand.get("city"),
                        state=cand.get("state"),
                    ),
                    overall_score=float(prob),
                    match_probability=float(prob),
                    field_scores=None,
                    match_type_used=MatchType.PROBABILISTIC,
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Splink scoring error: {e}")
            # Fallback to fuzzy scoring
            return self._score_deterministic_fuzzy(record, candidates)
    
    async def _score_hybrid(
        self,
        record: RecordInput,
        candidates: List[dict],
    ) -> List[MatchResult]:
        """
        Hybrid scoring: deterministic fuzzy + probabilistic.
        Combines both approaches for best accuracy.
        """
        # Get fuzzy scores
        fuzzy_results = self._score_deterministic_fuzzy(record, candidates)
        
        # Get probabilistic scores
        prob_results = await self._score_probabilistic(record, candidates)
        
        # Merge scores (weighted average)
        prob_map = {str(r.record.id): r for r in prob_results}
        
        merged = []
        for fuzzy in fuzzy_results:
            record_id = str(fuzzy.record.id)
            prob = prob_map.get(record_id)
            
            if prob:
                # Weighted combination: 40% fuzzy, 60% probabilistic
                combined_score = (fuzzy.overall_score * 0.4) + (prob.overall_score * 0.6)
                fuzzy.overall_score = combined_score
                fuzzy.match_probability = prob.match_probability
            
            fuzzy.match_type_used = MatchType.HYBRID
            merged.append(fuzzy)
        
        return merged


# Global singleton instance
entity_matcher = EntityMatcher()


async def get_matcher() -> EntityMatcher:
    """Dependency for getting the matcher instance."""
    if not entity_matcher._initialized:
        await entity_matcher.initialize()
    return entity_matcher
