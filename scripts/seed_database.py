#!/usr/bin/env python3
"""
Database Seeding Script
Loads synthetic data from CSV into PostgreSQL with optimized batch inserts.

Usage:
    python seed_database.py --input synthetic_persons.csv
    python seed_database.py --input persons.csv --batch-size 10000
"""

import argparse
import csv
import sys
import uuid
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm


def get_connection(database_url: str):
    """Create database connection."""
    return psycopg2.connect(database_url)


def create_tables(conn):
    """Create tables and extensions if they don't exist."""
    with conn.cursor() as cur:
        # Create extensions
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;")
        
        # Create persons table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                first_name VARCHAR(100) NOT NULL,
                middle_name VARCHAR(100),
                last_name VARCHAR(100) NOT NULL,
                birth_date DATE,
                death_date DATE,
                ssn_last4 VARCHAR(4),
                ssn_hash VARCHAR(64),
                city VARCHAR(100),
                state VARCHAR(2),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                source_system VARCHAR(50) DEFAULT 'synthetic'
            );
        """)
        
        # Create match_jobs table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS match_jobs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                status VARCHAR(20) DEFAULT 'pending',
                total_records INTEGER DEFAULT 0,
                processed_records INTEGER DEFAULT 0,
                matched_records INTEGER DEFAULT 0,
                original_filename VARCHAR(255),
                file_format VARCHAR(20),
                results_path VARCHAR(500),
                error_message TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                started_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                user_id VARCHAR(255)
            );
        """)
        
        conn.commit()
        print("Tables created successfully")


def create_indexes(conn):
    """Create optimized indexes for entity resolution."""
    indexes = [
        # B-tree indexes
        ("idx_persons_first_name", "CREATE INDEX IF NOT EXISTS idx_persons_first_name ON persons(first_name);"),
        ("idx_persons_last_name", "CREATE INDEX IF NOT EXISTS idx_persons_last_name ON persons(last_name);"),
        ("idx_persons_birth_date", "CREATE INDEX IF NOT EXISTS idx_persons_birth_date ON persons(birth_date);"),
        ("idx_persons_state", "CREATE INDEX IF NOT EXISTS idx_persons_state ON persons(state);"),
        ("idx_persons_ssn_last4", "CREATE INDEX IF NOT EXISTS idx_persons_ssn_last4 ON persons(ssn_last4);"),
        ("idx_persons_ssn_hash", "CREATE INDEX IF NOT EXISTS idx_persons_ssn_hash ON persons(ssn_hash);"),
        
        # Composite indexes for blocking
        ("idx_persons_blocking_name_dob", "CREATE INDEX IF NOT EXISTS idx_persons_blocking_name_dob ON persons(last_name, birth_date);"),
        ("idx_persons_blocking_state_name", "CREATE INDEX IF NOT EXISTS idx_persons_blocking_state_name ON persons(state, last_name);"),
        
        # Trigram indexes (GIN)
        ("idx_persons_first_name_trgm", "CREATE INDEX IF NOT EXISTS idx_persons_first_name_trgm ON persons USING GIN (first_name gin_trgm_ops);"),
        ("idx_persons_last_name_trgm", "CREATE INDEX IF NOT EXISTS idx_persons_last_name_trgm ON persons USING GIN (last_name gin_trgm_ops);"),
        ("idx_persons_city_trgm", "CREATE INDEX IF NOT EXISTS idx_persons_city_trgm ON persons USING GIN (city gin_trgm_ops);"),
        
        # Soundex indexes
        ("idx_persons_soundex_last", "CREATE INDEX IF NOT EXISTS idx_persons_soundex_last ON persons (soundex(last_name));"),
        ("idx_persons_soundex_first", "CREATE INDEX IF NOT EXISTS idx_persons_soundex_first ON persons (soundex(first_name));"),
    ]
    
    print("Creating indexes (this may take a while for large datasets)...")
    
    with conn.cursor() as cur:
        for name, sql in tqdm(indexes, desc="Creating indexes"):
            try:
                cur.execute(sql)
                conn.commit()
            except Exception as e:
                print(f"Warning: Index {name} failed: {e}")
                conn.rollback()
    
    print("Indexes created successfully")


def count_lines(filepath: str) -> int:
    """Count lines in a file efficiently."""
    with open(filepath, 'rb') as f:
        return sum(1 for _ in f) - 1  # Subtract header


def load_data(
    conn,
    input_file: str,
    batch_size: int = 5000,
    truncate: bool = False,
):
    """Load data from CSV into database."""
    if truncate:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE persons;")
            conn.commit()
            print("Truncated persons table")
    
    total_rows = count_lines(input_file)
    print(f"Loading {total_rows:,} records from {input_file}...")
    
    insert_sql = """
        INSERT INTO persons (
            id, first_name, middle_name, last_name, 
            birth_date, death_date, ssn_last4, ssn_hash, 
            city, state
        ) VALUES %s
    """
    
    batch = []
    inserted = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        with tqdm(total=total_rows, desc="Loading") as pbar:
            for row in reader:
                # Prepare record tuple
                record = (
                    str(uuid.uuid4()),
                    row.get('first_name') or None,
                    row.get('middle_name') or None,
                    row.get('last_name') or None,
                    row.get('birth_date') or None,
                    row.get('death_date') or None,
                    row.get('ssn_last4') or None,
                    row.get('ssn_hash') or None,
                    row.get('city') or None,
                    row.get('state') or None,
                )
                batch.append(record)
                
                if len(batch) >= batch_size:
                    with conn.cursor() as cur:
                        execute_values(cur, insert_sql, batch)
                    conn.commit()
                    inserted += len(batch)
                    pbar.update(len(batch))
                    batch = []
            
            # Insert remaining records
            if batch:
                with conn.cursor() as cur:
                    execute_values(cur, insert_sql, batch)
                conn.commit()
                inserted += len(batch)
                pbar.update(len(batch))
    
    print(f"\nInserted {inserted:,} records")


def analyze_table(conn):
    """Run ANALYZE on the persons table for query optimization."""
    print("Running ANALYZE on persons table...")
    with conn.cursor() as cur:
        cur.execute("ANALYZE persons;")
    conn.commit()
    print("ANALYZE complete")


def print_stats(conn):
    """Print database statistics."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM persons;")
        total = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM persons WHERE death_date IS NOT NULL;")
        with_death = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(DISTINCT state) FROM persons;")
        states = cur.fetchone()[0]
        
        cur.execute("SELECT pg_size_pretty(pg_total_relation_size('persons'));")
        size = cur.fetchone()[0]
    
    print("\n=== Database Statistics ===")
    print(f"Total records: {total:,}")
    print(f"Records with death date: {with_death:,} ({with_death/total*100:.1f}%)")
    print(f"Unique states: {states}")
    print(f"Table size (with indexes): {size}")


def main():
    parser = argparse.ArgumentParser(
        description="Load synthetic data into PostgreSQL"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input CSV file path"
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default="postgresql://postgres:postgres@localhost:5432/entity_resolution",
        help="PostgreSQL connection URL"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Records per batch insert (default: 5000)"
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate table before loading"
    )
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Skip index creation"
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Only create tables and indexes, don't load data"
    )
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not args.create_only and not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print(f"Connecting to database...")
    conn = get_connection(args.database_url)
    
    try:
        # Create tables
        create_tables(conn)
        
        if args.create_only:
            if not args.skip_indexes:
                create_indexes(conn)
            print("Database setup complete (tables and indexes created)")
            return
        
        # Load data
        load_data(
            conn,
            args.input,
            batch_size=args.batch_size,
            truncate=args.truncate,
        )
        
        # Create indexes after data load (more efficient)
        if not args.skip_indexes:
            create_indexes(conn)
        
        # Analyze for query optimization
        analyze_table(conn)
        
        # Print stats
        print_stats(conn)
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
