# Entity Resolution API

High-performance entity matching service supporting deterministic (exact/fuzzy) and probabilistic (Splink) matching algorithms against a PostgreSQL database.

## Features

- **Single Record Matching**: Real-time matching with sub-second response times
- **Bulk Upload Processing**: Support for CSV, JSON, Excel, Parquet, XML, and TXT files (up to 500 records)
- **Multiple Matching Algorithms**:
  - Deterministic Exact: Strict field matching
  - Deterministic Fuzzy: Levenshtein/similarity-based matching
  - Probabilistic: Splink-based Fellegi-Sunter model
  - Hybrid: Combines fuzzy blocking with probabilistic scoring (recommended)
- **Optimized for Scale**: Designed for 10M+ records with efficient blocking strategies
- **Modern API**: FastAPI with automatic OpenAPI documentation

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- PostgreSQL 16+ (or use Docker)

### 1. Clone and Setup

```bash
# Clone the repository
cd entity-resolution-api

# Copy environment file
cp .env.example .env

# Start services with Docker Compose
docker-compose up -d postgres redis
```

### 2. Generate Synthetic Data (Optional)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate 100K records for testing (adjust --count as needed)
python scripts/generate_synthetic_data.py --count 100000 --output data/persons.csv

# For full 10M records (takes ~30-60 minutes):
# python scripts/generate_synthetic_data.py --count 10000000 --output data/persons.csv
```

### 3. Seed the Database

```bash
# Load data into PostgreSQL
python scripts/seed_database.py --input data/persons.csv

# Or just create tables without data:
python scripts/seed_database.py --create-only
```

### 4. Start the API

```bash
# Option A: With Docker
docker-compose up api

# Option B: Locally
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/match/health

## API Usage

### Single Record Match

```bash
curl -X POST "http://localhost:8000/api/v1/match/match" \
  -H "Content-Type: application/json" \
  -d '{
    "record": {
      "first_name": "John",
      "last_name": "Smith",
      "birth_date": "1985-03-15",
      "state": "NY"
    },
    "match_type": "hybrid",
    "threshold": 0.7
  }'
```

### Bulk Upload

```bash
curl -X POST "http://localhost:8000/api/v1/bulk/upload" \
  -F "file=@records.csv" \
  -F "match_type=hybrid" \
  -F "threshold=0.7"
```

### Check Job Status

```bash
curl "http://localhost:8000/api/v1/bulk/job/{job_id}"
```

### Download Results

```bash
# JSON format
curl "http://localhost:8000/api/v1/bulk/job/{job_id}/results?format=json"

# CSV format
curl "http://localhost:8000/api/v1/bulk/job/{job_id}/results?format=csv" -o results.csv
```

## Project Structure

```
entity-resolution-api/
├── app/
│   ├── api/                 # API route handlers
│   │   ├── match.py         # Single record matching endpoints
│   │   └── bulk.py          # Bulk upload endpoints
│   ├── core/                # Core configuration
│   │   ├── config.py        # Settings management
│   │   └── database.py      # Database connections
│   ├── models/              # Data models
│   │   ├── database.py      # SQLAlchemy models
│   │   └── schemas.py       # Pydantic schemas
│   ├── services/            # Business logic
│   │   └── matcher.py       # Entity matching service
│   ├── utils/               # Utilities
│   │   └── file_parser.py   # File format parsers
│   └── main.py              # FastAPI application
├── scripts/
│   ├── generate_synthetic_data.py  # Data generator
│   ├── seed_database.py            # Database seeder
│   └── init-db.sql                 # SQL initialization
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Matching Algorithms

### Deterministic Exact
- Exact string comparison on all fields
- Fastest, but misses variations/typos
- Best for high-quality, standardized data

### Deterministic Fuzzy
- Uses Levenshtein distance, Jaro-Winkler similarity
- Handles typos and minor variations
- Good balance of speed and flexibility

### Probabilistic (Splink)
- Fellegi-Sunter probabilistic model
- Learns field weights from data patterns
- Best accuracy for complex matching

### Hybrid (Recommended)
- Uses fuzzy blocking for candidate generation
- Applies probabilistic scoring to candidates
- Best balance of performance and accuracy

## File Format Support

| Format | Extension | Notes |
|--------|-----------|-------|
| CSV | .csv | Auto-detects encoding |
| JSON | .json | Array or line-delimited |
| Excel | .xlsx, .xls | First sheet only |
| Text | .txt | Tab or pipe delimited |
| Parquet | .parquet | Efficient for large files |
| XML | .xml | Auto-detects record elements |

### Expected Columns

| Field | Required | Aliases |
|-------|----------|---------|
| first_name | Yes | firstname, fname, given_name |
| last_name | Yes | lastname, lname, surname |
| middle_name | No | middlename, mname |
| birth_date | No | dob, date_of_birth |
| ssn_last4 | No | ssn4, ssn (extracts last 4) |
| city | No | town |
| state | No | state_code, province |

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | PostgreSQL async connection URL |
| `MATCH_THRESHOLD` | 0.7 | Minimum score for matches |
| `MAX_RESULTS` | 100 | Max results per query |
| `MAX_BULK_RECORDS` | 500 | Max records per bulk upload |

See `.env.example` for full configuration options.

## Development Roadmap

- [x] **Phase 1**: Core matching engine and API
- [ ] **Phase 2**: Celery workers for async bulk processing
- [ ] **Phase 3**: OAuth authentication (Google/GitHub)
- [ ] **Phase 4**: Modern web frontend
- [ ] **Phase 5**: Production deployment (Neon + Vercel)

## Performance Notes

- Single record matching: 200-500ms (10M records)
- Bulk processing: ~2-5 records/second
- Database size: ~3-5GB for 10M records with indexes
- GIN trigram indexes significantly improve fuzzy search performance

## License

MIT License
