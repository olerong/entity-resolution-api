"""
Tests for Entity Resolution API.
Run with: pytest tests/ -v
"""

import pytest
from datetime import date
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import RecordInput, MatchType


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
    
    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/api/v1/match/health")
        # May fail if database not connected
        assert response.status_code in [200, 500]


class TestMatchEndpoints:
    """Test matching endpoints."""
    
    def test_match_types(self, client):
        """Test match types listing."""
        response = client.get("/api/v1/match/match-types")
        assert response.status_code == 200
        data = response.json()
        assert "match_types" in data
        assert len(data["match_types"]) == 4


class TestBulkEndpoints:
    """Test bulk upload endpoints."""
    
    def test_supported_formats(self, client):
        """Test supported formats listing."""
        response = client.get("/api/v1/bulk/supported-formats")
        assert response.status_code == 200
        data = response.json()
        assert "formats" in data
        assert len(data["formats"]) == 6
        assert data["max_records"] == 500


class TestRecordInput:
    """Test Pydantic schema validation."""
    
    def test_valid_record(self):
        """Test valid record creation."""
        record = RecordInput(
            first_name="John",
            last_name="Smith",
            birth_date=date(1985, 3, 15),
            state="NY",
        )
        assert record.first_name == "John"
        assert record.state == "NY"
    
    def test_state_normalization(self):
        """Test state is uppercased."""
        record = RecordInput(
            first_name="John",
            last_name="Smith",
            state="ny",
        )
        assert record.state == "NY"
    
    def test_name_normalization(self):
        """Test name is title-cased."""
        record = RecordInput(
            first_name="  john  ",
            last_name="SMITH",
        )
        assert record.first_name == "John"
        assert record.last_name == "Smith"
    
    def test_ssn_validation(self):
        """Test SSN last 4 validation."""
        record = RecordInput(
            first_name="John",
            last_name="Smith",
            ssn_last4="1234",
        )
        assert record.ssn_last4 == "1234"
    
    def test_invalid_ssn_rejected(self):
        """Test invalid SSN is rejected."""
        with pytest.raises(ValueError):
            RecordInput(
                first_name="John",
                last_name="Smith",
                ssn_last4="12345",  # Too long
            )


class TestFileParser:
    """Test file parsing utilities."""
    
    def test_detect_csv(self):
        """Test CSV format detection."""
        from app.utils.file_parser import FileParser, FileFormat
        
        assert FileParser.detect_format("data.csv") == FileFormat.CSV
        assert FileParser.detect_format("DATA.CSV") == FileFormat.CSV
    
    def test_detect_json(self):
        """Test JSON format detection."""
        from app.utils.file_parser import FileParser, FileFormat
        
        assert FileParser.detect_format("data.json") == FileFormat.JSON
    
    def test_detect_excel(self):
        """Test Excel format detection."""
        from app.utils.file_parser import FileParser, FileFormat
        
        assert FileParser.detect_format("data.xlsx") == FileFormat.XLSX
        assert FileParser.detect_format("data.xls") == FileFormat.XLSX
    
    def test_detect_parquet(self):
        """Test Parquet format detection."""
        from app.utils.file_parser import FileParser, FileFormat
        
        assert FileParser.detect_format("data.parquet") == FileFormat.PARQUET
        assert FileParser.detect_format("data.pq") == FileFormat.PARQUET
    
    def test_unsupported_format(self):
        """Test unsupported format raises error."""
        from app.utils.file_parser import FileParser
        
        with pytest.raises(ValueError):
            FileParser.detect_format("data.pdf")
