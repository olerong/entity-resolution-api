"""
File parsing utilities for bulk upload.
Supports: CSV, JSON, XLSX, TXT, Parquet, XML
"""

import io
import json
import logging
from typing import List, Dict, Any, BinaryIO
from pathlib import Path

import pandas as pd

from app.models.schemas import FileFormat, RecordInput

logger = logging.getLogger(__name__)


# Standard column name mappings
COLUMN_MAPPINGS = {
    # first_name variations
    "first_name": "first_name",
    "firstname": "first_name",
    "first": "first_name",
    "fname": "first_name",
    "given_name": "first_name",
    "givenname": "first_name",
    
    # middle_name variations
    "middle_name": "middle_name",
    "middlename": "middle_name",
    "middle": "middle_name",
    "mname": "middle_name",
    
    # last_name variations
    "last_name": "last_name",
    "lastname": "last_name",
    "last": "last_name",
    "lname": "last_name",
    "surname": "last_name",
    "family_name": "last_name",
    "familyname": "last_name",
    
    # birth_date variations
    "birth_date": "birth_date",
    "birthdate": "birth_date",
    "dob": "birth_date",
    "date_of_birth": "birth_date",
    "dateofbirth": "birth_date",
    "birthday": "birth_date",
    
    # ssn variations
    "ssn_last4": "ssn_last4",
    "ssn_last_4": "ssn_last4",
    "ssn4": "ssn_last4",
    "ssn": "ssn_last4",  # Will extract last 4
    "social": "ssn_last4",
    "social_security": "ssn_last4",
    
    # city variations
    "city": "city",
    "town": "city",
    "municipality": "city",
    
    # state variations
    "state": "state",
    "state_code": "state",
    "statecode": "state",
    "province": "state",
    "region": "state",
}


class FileParser:
    """Parse various file formats into standardized record format."""
    
    @staticmethod
    def detect_format(filename: str) -> FileFormat:
        """Detect file format from filename extension."""
        ext = Path(filename).suffix.lower().lstrip(".")
        
        format_map = {
            "csv": FileFormat.CSV,
            "json": FileFormat.JSON,
            "xlsx": FileFormat.XLSX,
            "xls": FileFormat.XLSX,
            "txt": FileFormat.TXT,
            "parquet": FileFormat.PARQUET,
            "pq": FileFormat.PARQUET,
            "xml": FileFormat.XML,
        }
        
        if ext not in format_map:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return format_map[ext]
    
    @staticmethod
    def parse(
        file_content: BinaryIO,
        file_format: FileFormat,
        filename: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Parse file content into list of record dictionaries.
        
        Args:
            file_content: File-like object with binary content
            file_format: Expected format
            filename: Original filename (for error messages)
        
        Returns:
            List of dictionaries with standardized field names
        """
        try:
            if file_format == FileFormat.CSV:
                return FileParser._parse_csv(file_content)
            elif file_format == FileFormat.JSON:
                return FileParser._parse_json(file_content)
            elif file_format == FileFormat.XLSX:
                return FileParser._parse_xlsx(file_content)
            elif file_format == FileFormat.TXT:
                return FileParser._parse_txt(file_content)
            elif file_format == FileFormat.PARQUET:
                return FileParser._parse_parquet(file_content)
            elif file_format == FileFormat.XML:
                return FileParser._parse_xml(file_content)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
        except Exception as e:
            logger.error(f"Failed to parse {filename}: {e}")
            raise ValueError(f"Failed to parse file: {str(e)}")
    
    @staticmethod
    def _parse_csv(file_content: BinaryIO) -> List[Dict[str, Any]]:
        """Parse CSV file."""
        # Try to detect encoding
        content = file_content.read()
        
        # Try UTF-8 first, then latin-1
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                text = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Unable to decode file with supported encodings")
        
        df = pd.read_csv(io.StringIO(text))
        return FileParser._standardize_dataframe(df)
    
    @staticmethod
    def _parse_json(file_content: BinaryIO) -> List[Dict[str, Any]]:
        """Parse JSON file (array of objects or line-delimited)."""
        content = file_content.read().decode("utf-8")
        
        # Try parsing as JSON array
        try:
            data = json.loads(content)
            if isinstance(data, list):
                df = pd.DataFrame(data)
                return FileParser._standardize_dataframe(df)
            elif isinstance(data, dict):
                # Single record
                df = pd.DataFrame([data])
                return FileParser._standardize_dataframe(df)
        except json.JSONDecodeError:
            pass
        
        # Try line-delimited JSON
        records = []
        for line in content.strip().split("\n"):
            if line.strip():
                records.append(json.loads(line))
        
        df = pd.DataFrame(records)
        return FileParser._standardize_dataframe(df)
    
    @staticmethod
    def _parse_xlsx(file_content: BinaryIO) -> List[Dict[str, Any]]:
        """Parse Excel file."""
        df = pd.read_excel(file_content, engine="openpyxl")
        return FileParser._standardize_dataframe(df)
    
    @staticmethod
    def _parse_txt(file_content: BinaryIO) -> List[Dict[str, Any]]:
        """Parse text file (tab-delimited or pipe-delimited)."""
        content = file_content.read().decode("utf-8")
        
        # Detect delimiter
        first_line = content.split("\n")[0]
        if "\t" in first_line:
            delimiter = "\t"
        elif "|" in first_line:
            delimiter = "|"
        else:
            delimiter = ","  # Fallback to comma
        
        df = pd.read_csv(io.StringIO(content), delimiter=delimiter)
        return FileParser._standardize_dataframe(df)
    
    @staticmethod
    def _parse_parquet(file_content: BinaryIO) -> List[Dict[str, Any]]:
        """Parse Parquet file."""
        df = pd.read_parquet(file_content)
        return FileParser._standardize_dataframe(df)
    
    @staticmethod
    def _parse_xml(file_content: BinaryIO) -> List[Dict[str, Any]]:
        """Parse XML file."""
        import xml.etree.ElementTree as ET
        
        content = file_content.read()
        root = ET.fromstring(content)
        
        records = []
        
        # Try to find record elements
        # Common patterns: /records/record, /data/row, /root/item
        for tag in ["record", "row", "item", "person", "entry"]:
            elements = root.findall(f".//{tag}")
            if elements:
                for elem in elements:
                    record = {}
                    for child in elem:
                        record[child.tag] = child.text
                    records.append(record)
                break
        
        if not records:
            # Fallback: treat each direct child as a record
            for child in root:
                record = {}
                for subchild in child:
                    record[subchild.tag] = subchild.text
                records.append(record)
        
        df = pd.DataFrame(records)
        return FileParser._standardize_dataframe(df)
    
    @staticmethod
    def _standardize_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Standardize column names and clean data."""
        # Lowercase all column names
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # Rename columns to standard names
        rename_map = {}
        for col in df.columns:
            if col in COLUMN_MAPPINGS:
                rename_map[col] = COLUMN_MAPPINGS[col]
        
        df = df.rename(columns=rename_map)
        
        # Process each record
        records = []
        for _, row in df.iterrows():
            record = {}
            
            # Extract required fields
            for field in ["first_name", "middle_name", "last_name", 
                         "birth_date", "ssn_last4", "city", "state"]:
                if field in row and pd.notna(row[field]):
                    value = row[field]
                    
                    # Special handling for SSN
                    if field == "ssn_last4":
                        value = FileParser._extract_ssn_last4(value)
                    # Special handling for birth_date
                    elif field == "birth_date":
                        value = FileParser._parse_date(value)
                    # Special handling for state
                    elif field == "state":
                        value = str(value).upper()[:2] if value else None
                    else:
                        value = str(value).strip() if value else None
                    
                    record[field] = value
            
            # Only include records with at least first and last name
            if record.get("first_name") and record.get("last_name"):
                records.append(record)
        
        return records
    
    @staticmethod
    def _extract_ssn_last4(value) -> str:
        """Extract last 4 digits from SSN value."""
        if pd.isna(value):
            return None
        
        # Remove non-digits
        digits = "".join(c for c in str(value) if c.isdigit())
        
        if len(digits) >= 4:
            return digits[-4:]
        elif len(digits) > 0:
            return digits.zfill(4)  # Pad with zeros if needed
        
        return None
    
    @staticmethod
    def _parse_date(value) -> str:
        """Parse date value to ISO format string."""
        if pd.isna(value):
            return None
        
        try:
            # If already a datetime
            if hasattr(value, "isoformat"):
                return value.date().isoformat() if hasattr(value, "date") else value.isoformat()
            
            # Try pandas parsing
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.notna(parsed):
                return parsed.date().isoformat()
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def validate_records(records: List[Dict[str, Any]], max_records: int = 500) -> tuple:
        """
        Validate parsed records.
        
        Returns:
            Tuple of (valid_records, errors)
        """
        if len(records) > max_records:
            raise ValueError(f"Too many records: {len(records)} exceeds maximum of {max_records}")
        
        valid = []
        errors = []
        
        for i, record in enumerate(records):
            row_num = i + 1
            try:
                # Validate using Pydantic model
                validated = RecordInput(**record)
                valid.append({
                    "row_number": row_num,
                    "record": validated,
                })
            except Exception as e:
                errors.append({
                    "row_number": row_num,
                    "error": str(e),
                    "data": record,
                })
        
        return valid, errors
