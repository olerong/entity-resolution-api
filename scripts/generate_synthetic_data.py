#!/usr/bin/env python3
"""
Synthetic Data Generator for Entity Resolution Testing.
Generates realistic person records with controlled data quality issues.

Usage:
    python generate_synthetic_data.py --count 10000000 --output persons.csv
    python generate_synthetic_data.py --count 100000 --output sample.csv  # For testing
"""

import argparse
import csv
import hashlib
import random
import string
from datetime import date, timedelta
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Optional, Tuple
import sys

from faker import Faker
from tqdm import tqdm

# Initialize Faker with US locale
fake = Faker('en_US')
Faker.seed(42)
random.seed(42)

# Nickname mappings for realistic variations
NICKNAMES = {
    "william": ["will", "bill", "billy", "liam"],
    "robert": ["rob", "bob", "bobby", "bert"],
    "richard": ["rich", "rick", "dick", "ricky"],
    "james": ["jim", "jimmy", "jamie"],
    "john": ["jack", "johnny", "jon"],
    "michael": ["mike", "mikey", "mick"],
    "elizabeth": ["liz", "lizzy", "beth", "betty", "eliza"],
    "jennifer": ["jen", "jenny", "jenn"],
    "margaret": ["maggie", "meg", "peggy", "marge"],
    "catherine": ["cathy", "kate", "katie", "cat"],
    "patricia": ["pat", "patty", "trish"],
    "joseph": ["joe", "joey"],
    "thomas": ["tom", "tommy"],
    "daniel": ["dan", "danny"],
    "christopher": ["chris", "topher"],
    "matthew": ["matt", "matty"],
    "anthony": ["tony"],
    "steven": ["steve"],
    "david": ["dave", "davey"],
    "andrew": ["andy", "drew"],
    "alexander": ["alex", "xander"],
    "nicholas": ["nick", "nicky"],
    "samantha": ["sam", "sammy"],
    "katherine": ["kate", "kathy", "katie"],
    "rebecca": ["becca", "becky"],
    "jessica": ["jess", "jessie"],
    "amanda": ["mandy"],
    "stephanie": ["steph"],
    "victoria": ["vicky", "tori"],
    "benjamin": ["ben", "benny"],
    "jonathan": ["jon", "jonny"],
    "theodore": ["ted", "teddy", "theo"],
    "edward": ["ed", "eddie", "ted"],
    "charles": ["charlie", "chuck"],
}

# Common typo patterns
TYPO_PATTERNS = [
    lambda s: s[0] + s[2] + s[1] + s[3:] if len(s) > 3 else s,  # Swap chars
    lambda s: s.replace('a', 'e') if 'a' in s else s,  # Vowel swap
    lambda s: s.replace('i', 'e') if 'i' in s else s,
    lambda s: s + s[-1] if s else s,  # Double last letter
    lambda s: s[:-1] if len(s) > 2 else s,  # Drop last letter
]


def generate_ssn_last4() -> str:
    """Generate synthetic SSN last 4 digits."""
    return f"{random.randint(0, 9999):04d}"


def generate_ssn_hash(last4: str) -> str:
    """Generate a hash for SSN (simulating full SSN storage)."""
    # Use advertising range 987-XX-XXXX to ensure not real
    fake_full = f"987-{random.randint(10, 99)}-{last4}"
    return hashlib.sha256(fake_full.encode()).hexdigest()


def introduce_typo(name: str) -> str:
    """Introduce a realistic typo into a name."""
    if not name or len(name) < 3:
        return name
    pattern = random.choice(TYPO_PATTERNS)
    return pattern(name)


def get_nickname(name: str) -> Optional[str]:
    """Get a nickname for a given name."""
    name_lower = name.lower()
    if name_lower in NICKNAMES:
        return random.choice(NICKNAMES[name_lower]).title()
    return None


def generate_birth_date(
    min_age: int = 18,
    max_age: int = 95,
) -> date:
    """Generate a realistic birth date."""
    today = date.today()
    days_old = random.randint(min_age * 365, max_age * 365)
    return today - timedelta(days=days_old)


def generate_death_date(birth_date: date, alive_probability: float = 0.92) -> Optional[date]:
    """
    Generate death date based on birth date.
    ~8% of records will have a death date.
    """
    if random.random() < alive_probability:
        return None
    
    # Age at death: normal distribution around 75
    age_at_death = int(random.gauss(75, 12))
    age_at_death = max(18, min(100, age_at_death))  # Clamp
    
    death_date = birth_date + timedelta(days=age_at_death * 365 + random.randint(0, 364))
    
    # Don't return future dates
    if death_date > date.today():
        return None
    
    return death_date


def generate_person(
    introduce_errors: bool = True,
    error_rate: float = 0.03,  # 3% error rate
    missing_rate: float = 0.05,  # 5% missing data rate
) -> Dict:
    """Generate a single person record."""
    first_name = fake.first_name()
    middle_name = fake.first_name() if random.random() > 0.3 else None
    last_name = fake.last_name()
    birth_date = generate_birth_date()
    death_date = generate_death_date(birth_date)
    ssn_last4 = generate_ssn_last4()
    city = fake.city()
    state = fake.state_abbr()
    
    # Introduce data quality issues
    if introduce_errors:
        # Typos (3%)
        if random.random() < error_rate:
            if random.random() < 0.5:
                first_name = introduce_typo(first_name)
            else:
                last_name = introduce_typo(last_name)
        
        # Nicknames (10%)
        if random.random() < 0.10:
            nickname = get_nickname(first_name)
            if nickname:
                first_name = nickname
        
        # Missing middle name (20% additional)
        if middle_name and random.random() < 0.20:
            middle_name = None
        
        # Missing SSN (15%)
        if random.random() < 0.15:
            ssn_last4 = None
        
        # Missing city (8%)
        if random.random() < 0.08:
            city = None
    
    return {
        "first_name": first_name,
        "middle_name": middle_name,
        "last_name": last_name,
        "birth_date": birth_date.isoformat(),
        "death_date": death_date.isoformat() if death_date else None,
        "ssn_last4": ssn_last4,
        "ssn_hash": generate_ssn_hash(ssn_last4) if ssn_last4 else None,
        "city": city,
        "state": state,
    }


def generate_chunk(args: Tuple[int, int, bool]) -> List[Dict]:
    """Generate a chunk of records (for multiprocessing)."""
    chunk_id, chunk_size, introduce_errors = args
    
    # Re-seed for this chunk to ensure reproducibility
    Faker.seed(42 + chunk_id)
    random.seed(42 + chunk_id)
    
    return [generate_person(introduce_errors) for _ in range(chunk_size)]


def generate_dataset(
    count: int,
    output_file: str,
    num_workers: int = None,
    chunk_size: int = 10000,
    introduce_errors: bool = True,
):
    """
    Generate a large synthetic dataset.
    
    Args:
        count: Number of records to generate
        output_file: Output CSV file path
        num_workers: Number of parallel workers (default: CPU count)
        chunk_size: Records per chunk
        introduce_errors: Whether to introduce data quality issues
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    # Calculate chunks
    num_chunks = (count + chunk_size - 1) // chunk_size
    chunk_args = [
        (i, min(chunk_size, count - i * chunk_size), introduce_errors)
        for i in range(num_chunks)
    ]
    
    print(f"Generating {count:,} records with {num_workers} workers...")
    print(f"Output file: {output_file}")
    
    # CSV header
    fieldnames = [
        "first_name", "middle_name", "last_name",
        "birth_date", "death_date",
        "ssn_last4", "ssn_hash",
        "city", "state"
    ]
    
    records_written = 0
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        with Pool(num_workers) as pool:
            with tqdm(total=count, desc="Generating") as pbar:
                for chunk in pool.imap(generate_chunk, chunk_args):
                    writer.writerows(chunk)
                    records_written += len(chunk)
                    pbar.update(len(chunk))
    
    print(f"\nGenerated {records_written:,} records")
    print(f"File size: {get_file_size(output_file)}")


def get_file_size(filepath: str) -> str:
    """Get human-readable file size."""
    import os
    size = os.path.getsize(filepath)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic person data for entity resolution testing"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=100000,
        help="Number of records to generate (default: 100000)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="synthetic_persons.csv",
        help="Output CSV file path (default: synthetic_persons.csv)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Records per chunk (default: 10000)"
    )
    parser.add_argument(
        "--no-errors",
        action="store_true",
        help="Generate clean data without quality issues"
    )
    
    args = parser.parse_args()
    
    generate_dataset(
        count=args.count,
        output_file=args.output,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
        introduce_errors=not args.no_errors,
    )


if __name__ == "__main__":
    main()
