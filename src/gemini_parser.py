"""
Gemini AI Document Parser
Uses Google's Gemini API to parse unstructured documents (PDFs, images, etc.)
and extract credit portfolio data.
"""

import os
import json
import base64
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import PyPDF2
import io


class GeminiDocumentParser:
    """Parse credit portfolio documents using Gemini AI."""

    # Expected fields for credit portfolio analysis
    EXPECTED_FIELDS = [
        'account_id', 'credit_limit', 'balance', 'payment_status',
        'age', 'gender', 'education', 'marital_status',
        'payment_history_1', 'payment_history_2', 'payment_history_3',
        'payment_history_4', 'payment_history_5', 'payment_history_6',
        'bill_amount_1', 'bill_amount_2', 'bill_amount_3',
        'bill_amount_4', 'bill_amount_5', 'bill_amount_6',
        'payment_amount_1', 'payment_amount_2', 'payment_amount_3',
        'payment_amount_4', 'payment_amount_5', 'payment_amount_6',
        'default_status'
    ]

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini parser.

        Args:
            api_key: Gemini API key. If None, reads from environment.
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def parse_document(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Parse any document type and extract credit portfolio data.

        Args:
            file_path: Path to document (PDF, image, etc.)

        Returns:
            Tuple of (DataFrame with parsed data, metadata dict)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type and parse accordingly
        ext = file_path.suffix.lower()

        if ext == '.pdf':
            return self._parse_pdf(file_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            return self._parse_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _parse_pdf(self, file_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Parse PDF document using Gemini."""
        # Extract text from PDF
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)

            # Extract text from all pages
            text_content = []
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text_content.append(page.extract_text())

            full_text = "\n\n".join(text_content)

        # If PDF has images or tables, read the actual file for multimodal parsing
        with open(file_path, 'rb') as f:
            pdf_data = f.read()

        # Use Gemini to parse the content
        return self._extract_with_gemini(
            content=full_text,
            file_data=pdf_data,
            file_type='application/pdf',
            metadata={'source': str(file_path), 'pages': num_pages}
        )

    def _parse_image(self, file_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Parse image document using Gemini vision."""
        # Load image
        image = Image.open(file_path)

        # Read file data
        with open(file_path, 'rb') as f:
            image_data = f.read()

        # Use Gemini to parse the image
        return self._extract_with_gemini(
            content=None,
            file_data=image_data,
            file_type=f'image/{file_path.suffix[1:]}',
            metadata={'source': str(file_path), 'size': image.size}
        )

    def _extract_with_gemini(
        self,
        content: Optional[str],
        file_data: bytes,
        file_type: str,
        metadata: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Use Gemini to extract structured credit portfolio data.

        Args:
            content: Text content (if available)
            file_data: Raw file bytes for multimodal parsing
            file_type: MIME type
            metadata: Additional metadata

        Returns:
            Tuple of (DataFrame, metadata)
        """
        # Create extraction prompt
        prompt = self._create_extraction_prompt()

        # Prepare multimodal content
        parts = [prompt]

        # Add file data for vision/document understanding
        if file_data:
            parts.append({
                'mime_type': file_type,
                'data': base64.b64encode(file_data).decode('utf-8')
            })

        # If we have text content, add it
        if content:
            parts.append(f"\n\nExtracted Text Content:\n{content[:10000]}")  # Limit to 10K chars

        try:
            # Generate response
            response = self.model.generate_content(parts)

            # Parse JSON response
            response_text = response.text.strip()

            # Extract JSON from markdown code blocks if present
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()

            parsed_data = json.loads(response_text)

            # Convert to DataFrame
            if 'records' in parsed_data:
                df = pd.DataFrame(parsed_data['records'])
            else:
                # Single record
                df = pd.DataFrame([parsed_data])

            # Update metadata
            metadata.update({
                'parser': 'gemini',
                'model': 'gemini-1.5-flash',
                'fields_found': list(df.columns),
                'record_count': len(df)
            })

            return df, metadata

        except Exception as e:
            # If Gemini fails, return empty DataFrame with error metadata
            metadata.update({
                'parser': 'gemini',
                'error': str(e),
                'success': False
            })
            return pd.DataFrame(), metadata

    def _create_extraction_prompt(self) -> str:
        """Create prompt for Gemini to extract credit portfolio data."""
        return f"""You are a financial document parser specializing in credit portfolio data extraction.

Your task is to analyze the provided document (PDF, image, or text) and extract credit account information.

EXPECTED DATA STRUCTURE:
Extract ALL available credit account records from the document. For each account, try to identify these fields:

- account_id: Account or customer identifier
- credit_limit: Maximum credit limit or loan amount
- balance: Current outstanding balance
- payment_status: Payment status (e.g., current, delinquent, paid, 0-6 for months past due)
- age: Customer age
- gender: Customer gender (M/F or Male/Female)
- education: Education level
- marital_status: Marital status
- payment_history_1 through payment_history_6: Payment history for last 6 months
- bill_amount_1 through bill_amount_6: Bill amounts for last 6 months
- payment_amount_1 through payment_amount_6: Payment amounts for last 6 months
- default_status: Default flag (0=no default, 1=default)

IMPORTANT INSTRUCTIONS:
1. If the document contains a TABLE, extract ALL rows as separate records
2. If it's a single account statement, extract that one account
3. Map the document's field names to the expected fields above (use your best judgment)
4. If a field is not found, omit it (don't use null)
5. Convert all numeric values to numbers (not strings)
6. For payment status: try to map to 0-6 scale where 0=current, 1=1 month late, etc.

OUTPUT FORMAT:
Return ONLY valid JSON in this exact format:

For multiple records (tables):
{{
  "records": [
    {{"account_id": "...", "credit_limit": 50000, "balance": 15000, ...}},
    {{"account_id": "...", "credit_limit": 30000, "balance": 8000, ...}}
  ]
}}

For single record (individual statement):
{{
  "account_id": "...",
  "credit_limit": 50000,
  "balance": 15000,
  ...
}}

Now analyze the document and extract the credit portfolio data:"""

    def validate_and_enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate extracted data and enrich with missing fields.

        Args:
            df: DataFrame with extracted data

        Returns:
            Validated and enriched DataFrame
        """
        if df.empty:
            return df

        # Ensure numeric types for numeric fields
        numeric_fields = [
            'credit_limit', 'balance', 'age',
            'payment_history_1', 'payment_history_2', 'payment_history_3',
            'payment_history_4', 'payment_history_5', 'payment_history_6',
            'bill_amount_1', 'bill_amount_2', 'bill_amount_3',
            'bill_amount_4', 'bill_amount_5', 'bill_amount_6',
            'payment_amount_1', 'payment_amount_2', 'payment_amount_3',
            'payment_amount_4', 'payment_amount_5', 'payment_amount_6',
            'default_status', 'payment_status'
        ]

        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')

        # Add missing expected fields
        for field in self.EXPECTED_FIELDS:
            if field not in df.columns:
                df[field] = None

        # Calculate utilization if we have the data
        if 'credit_limit' in df.columns and 'balance' in df.columns:
            df['utilization'] = (df['balance'] / df['credit_limit']).fillna(0)

        return df


def test_gemini_parser():
    """Test the Gemini parser with a sample."""
    parser = GeminiDocumentParser()
    print("✅ Gemini parser initialized successfully!")
    print(f"Model: {parser.model}")
    return parser


if __name__ == "__main__":
    test_gemini_parser()
