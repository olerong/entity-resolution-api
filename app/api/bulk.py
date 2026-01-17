"""
API routes for bulk file upload and matching.
"""

import io
import json
import csv
import logging
from typing import Optional
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.core.config import get_settings
from app.core.database import get_async_session
from app.models.database import MatchJob
from app.models.schemas import (
    MatchType, JobStatus, FileFormat, JobResponse,
    BulkMatchRequest, BulkMatchResponse, BulkRecordResult
)
from app.services.matcher import EntityMatcher, get_matcher, MatchConfig
from app.utils.file_parser import FileParser

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


@router.post("/upload", response_model=JobResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    match_type: MatchType = MatchType.HYBRID,
    threshold: Optional[float] = None,
    max_results_per_record: int = 10,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Upload a file for bulk matching.
    
    Supported formats: CSV, JSON, XLSX, TXT, Parquet, XML
    Maximum 500 records per file.
    
    Returns a job ID for tracking progress.
    """
    # Detect file format
    try:
        file_format = FileParser.detect_format(file.filename)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    
    # Read and parse file
    try:
        content = await file.read()
        records = FileParser.parse(
            io.BytesIO(content),
            file_format,
            file.filename,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse file: {str(e)}",
        )
    
    # Validate records
    try:
        valid_records, parse_errors = FileParser.validate_records(
            records,
            max_records=settings.max_bulk_records,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    
    if not valid_records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No valid records found. Errors: {parse_errors[:5]}",
        )
    
    # Create job record
    job = MatchJob(
        status=JobStatus.PENDING,
        total_records=len(valid_records),
        original_filename=file.filename,
        file_format=file_format.value,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    
    # Store records for processing (in real app, use Redis or temp file)
    # For now, we'll process in background task
    background_tasks.add_task(
        process_bulk_job,
        job_id=job.id,
        records=valid_records,
        match_type=match_type,
        threshold=threshold or settings.match_threshold,
        max_results=max_results_per_record,
    )
    
    return JobResponse(
        job_id=job.id,
        status=JobStatus.PENDING,
        total_records=job.total_records,
        processed_records=0,
        matched_records=0,
        progress_percent=0.0,
        created_at=job.created_at,
    )


async def process_bulk_job(
    job_id: UUID,
    records: list,
    match_type: MatchType,
    threshold: float,
    max_results: int,
):
    """Background task to process bulk matching job."""
    from app.core.database import AsyncSessionLocal, get_sync_session
    
    async with AsyncSessionLocal() as session:
        # Update job status to processing
        job = await session.get(MatchJob, job_id)
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow()
        await session.commit()
        
        try:
            matcher = await get_matcher()
            config = MatchConfig(
                match_type=match_type,
                threshold=threshold,
                max_results=max_results,
                include_scores=True,
            )
            
            results = []
            matched_count = 0
            
            for item in records:
                row_num = item["row_number"]
                record = item["record"]
                
                # Match this record
                match_result = await matcher.match_single(
                    session=session,
                    record=record,
                    config=config,
                )
                
                result = BulkRecordResult(
                    row_number=row_num,
                    input_record=record.model_dump(),
                    matches=[m.model_dump() for m in match_result.matches],
                    match_count=len(match_result.matches),
                    best_score=match_result.matches[0].overall_score if match_result.matches else None,
                )
                results.append(result)
                
                if match_result.matches:
                    matched_count += 1
                
                # Update progress periodically
                job.processed_records = row_num
                job.matched_records = matched_count
                await session.commit()
            
            # Save results (in production, save to file storage)
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.results_path = f"/tmp/job_{job_id}_results.json"
            
            # Write results to temp file
            with open(job.results_path, "w") as f:
                json.dump([r.model_dump() for r in results], f)
            
            await session.commit()
            
        except Exception as e:
            logger.error(f"Bulk job {job_id} failed: {e}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            await session.commit()


@router.get("/job/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Get the status of a bulk matching job."""
    job = await session.get(MatchJob, job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    
    progress = (job.processed_records / job.total_records * 100) if job.total_records > 0 else 0
    
    return JobResponse(
        job_id=job.id,
        status=JobStatus(job.status),
        total_records=job.total_records,
        processed_records=job.processed_records,
        matched_records=job.matched_records,
        progress_percent=progress,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
    )


@router.get("/job/{job_id}/results")
async def get_job_results(
    job_id: UUID,
    format: str = "json",
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get the results of a completed bulk matching job.
    
    Format options: json, csv
    """
    job = await session.get(MatchJob, job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job not completed. Current status: {job.status}",
        )
    
    # Load results from file
    try:
        with open(job.results_path, "r") as f:
            results = json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Results file not found",
        )
    
    if format.lower() == "csv":
        return _results_to_csv(results, job_id)
    else:
        return {
            "job_id": str(job_id),
            "total_records": job.total_records,
            "records_with_matches": job.matched_records,
            "results": results,
        }


def _results_to_csv(results: list, job_id: UUID) -> StreamingResponse:
    """Convert results to CSV format."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        "row_number", "input_first_name", "input_last_name", "input_birth_date",
        "match_count", "best_score",
        "match_1_id", "match_1_first_name", "match_1_last_name", "match_1_score",
        "match_2_id", "match_2_first_name", "match_2_last_name", "match_2_score",
        "match_3_id", "match_3_first_name", "match_3_last_name", "match_3_score",
    ])
    
    for r in results:
        row = [
            r["row_number"],
            r["input_record"].get("first_name", ""),
            r["input_record"].get("last_name", ""),
            r["input_record"].get("birth_date", ""),
            r["match_count"],
            r.get("best_score", ""),
        ]
        
        # Add up to 3 matches
        for i in range(3):
            if i < len(r["matches"]):
                m = r["matches"][i]
                rec = m.get("record", {})
                row.extend([
                    rec.get("id", ""),
                    rec.get("first_name", ""),
                    rec.get("last_name", ""),
                    m.get("overall_score", ""),
                ])
            else:
                row.extend(["", "", "", ""])
        
        writer.writerow(row)
    
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=matches_{job_id}.csv"
        },
    )


@router.delete("/job/{job_id}")
async def delete_job(
    job_id: UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Delete a job and its results."""
    job = await session.get(MatchJob, job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    
    # Delete results file if exists
    if job.results_path:
        import os
        try:
            os.remove(job.results_path)
        except FileNotFoundError:
            pass
    
    await session.delete(job)
    await session.commit()
    
    return {"message": "Job deleted", "job_id": str(job_id)}


@router.get("/supported-formats")
async def list_supported_formats():
    """List supported file formats for bulk upload."""
    return {
        "formats": [
            {"extension": ".csv", "name": "CSV", "mime_type": "text/csv"},
            {"extension": ".json", "name": "JSON", "mime_type": "application/json"},
            {"extension": ".xlsx", "name": "Excel", "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
            {"extension": ".txt", "name": "Text (tab/pipe delimited)", "mime_type": "text/plain"},
            {"extension": ".parquet", "name": "Parquet", "mime_type": "application/octet-stream"},
            {"extension": ".xml", "name": "XML", "mime_type": "application/xml"},
        ],
        "max_records": settings.max_bulk_records,
        "required_fields": ["first_name", "last_name"],
        "optional_fields": ["middle_name", "birth_date", "ssn_last4", "city", "state"],
    }
