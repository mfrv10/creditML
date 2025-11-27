"""
Smart Credit File Parser
Automatically detects and standardizes various credit file formats
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from typing import Dict, List, Tuple
import re

class CreditFileParser:
    """
    Intelligent parser that understands different credit file formats
    and automatically maps columns to standardized names
    """
    
    def __init__(self):
        # Standard column mappings (fuzzy matching dictionary)
        self.column_patterns = {
            'account_id': ['id', 'account_id', 'customer_id', 'client_id', 'acct_id', 'account'],
            'credit_limit': ['limit_bal', 'credit_limit', 'limit', 'max_credit', 'total_limit'],
            'sex': ['sex', 'gender'],
            'education': ['education', 'edu', 'education_level'],
            'marriage': ['marriage', 'marital_status', 'married'],
            'age': ['age'],
            'pay_status_0': ['pay_0', 'pay_status_0', 'payment_status', 'recent_payment'],
            'pay_status_2': ['pay_2', 'pay_status_2'],
            'pay_status_3': ['pay_3', 'pay_status_3'],
            'pay_status_4': ['pay_4', 'pay_status_4'],
            'pay_status_5': ['pay_5', 'pay_status_5'],
            'pay_status_6': ['pay_6', 'pay_status_6'],
            'bill_amount_1': ['bill_amt1', 'bill_amount_1', 'balance_1', 'statement_1'],
            'bill_amount_2': ['bill_amt2', 'bill_amount_2', 'balance_2', 'statement_2'],
            'bill_amount_3': ['bill_amt3', 'bill_amount_3', 'balance_3', 'statement_3'],
            'bill_amount_4': ['bill_amt4', 'bill_amount_4', 'balance_4', 'statement_4'],
            'bill_amount_5': ['bill_amt5', 'bill_amount_5', 'balance_5', 'statement_5'],
            'bill_amount_6': ['bill_amt6', 'bill_amount_6', 'balance_6', 'statement_6'],
            'payment_amount_1': ['pay_amt1', 'payment_1', 'payment_amt_1'],
            'payment_amount_2': ['pay_amt2', 'payment_2', 'payment_amt_2'],
            'payment_amount_3': ['pay_amt3', 'payment_3', 'payment_amt_3'],
            'payment_amount_4': ['pay_amt4', 'payment_4', 'payment_amt_4'],
            'payment_amount_5': ['pay_amt5', 'payment_5', 'payment_amt_5'],
            'payment_amount_6': ['pay_amt6', 'payment_6', 'payment_amt_6'],
            'default_flag': ['default', 'default_flag', 'default.payment.next.month', 'defaulted']
        }
    
    def analyze_file(self, file) -> Dict:
        """
        Analyze uploaded file and detect structure
        
        Returns:
            Dictionary with file metadata and structure info
        """
        # Load file with error handling
        if hasattr(file, 'name') and file.name.endswith('.csv'):
            df = pd.read_csv(file, on_bad_lines='skip', low_memory=False)
        elif isinstance(file, str) and file.endswith('.csv'):
            df = pd.read_csv(file, on_bad_lines='skip', low_memory=False)
        else:
            df = pd.read_excel(file)
        
        # Detect file type
        file_type = self._detect_file_type(df)
        
        # Map columns
        column_mapping = self._map_columns(df.columns)
        
        # Calculate data quality score
        quality_score = self._calculate_quality_score(df)
        
        # Create mapping dataframe for display
        mapping_df = pd.DataFrame([
            {'Original Column': orig, 'Mapped To': mapped, 'Confidence': conf}
            for orig, mapped, conf in column_mapping
        ])
        
        return {
            'file_type': file_type,
            'num_records': len(df),
            'num_columns': len(df.columns),
            'quality_score': quality_score,
            'column_mapping': mapping_df,
            'original_columns': list(df.columns)
        }
    
    def load_and_standardize(self, file) -> pd.DataFrame:
        """
        Load file and return standardized DataFrame
        """
        # Reset file pointer to beginning (in case it was read before)
        if hasattr(file, 'seek'):
            file.seek(0)
        
        # Load file with error handling
        if hasattr(file, 'name') and file.name.endswith('.csv'):
            df = pd.read_csv(file, on_bad_lines='skip', low_memory=False)
        elif isinstance(file, str) and file.endswith('.csv'):
            df = pd.read_csv(file, on_bad_lines='skip', low_memory=False)
        else:
            df = pd.read_excel(file)
        
        # Get column mapping
        column_mapping = self._map_columns(df.columns)
        
        # Create rename dictionary
        rename_dict = {orig: mapped for orig, mapped, conf in column_mapping if conf > 60}
        
        # Rename columns
        df_standardized = df.rename(columns=rename_dict)
        
        # Ensure required columns exist
        df_standardized = self._ensure_required_columns(df_standardized)
        
        # Clean and validate data
        df_standardized = self._clean_data(df_standardized)
        
        return df_standardized
    
    def _detect_file_type(self, df: pd.DataFrame) -> str:
        """Detect what type of credit file this is"""
        columns_lower = [c.lower() for c in df.columns]
        
        # Check for credit card indicators
        credit_card_indicators = ['pay_', 'bill_amt', 'limit_bal']
        if any(any(ind in col for ind in credit_card_indicators) for col in columns_lower):
            return "Credit Card Portfolio"
        
        # Check for loan indicators
        loan_indicators = ['loan_amount', 'interest_rate', 'maturity']
        if any(ind in ' '.join(columns_lower) for ind in loan_indicators):
            return "Loan Portfolio"
        
        # Default
        return "Credit Portfolio"
    
    def _map_columns(self, columns: List[str]) -> List[Tuple[str, str, int]]:
        """
        Map original columns to standardized names using fuzzy matching
        
        Returns:
            List of (original_name, mapped_name, confidence_score) tuples
        """
        mappings = []
        
        for col in columns:
            col_lower = col.lower().replace('_', '').replace('.', '')
            best_match = None
            best_score = 0
            
            # Try exact and fuzzy matching
            for standard_name, patterns in self.column_patterns.items():
                for pattern in patterns:
                    pattern_clean = pattern.lower().replace('_', '')
                    
                    # Exact match
                    if col_lower == pattern_clean:
                        best_match = standard_name
                        best_score = 100
                        break
                    
                    # Fuzzy match
                    score = fuzz.ratio(col_lower, pattern_clean)
                    if score > best_score:
                        best_score = score
                        best_match = standard_name
                
                if best_score == 100:
                    break
            
            if best_match and best_score > 60:
                mappings.append((col, best_match, best_score))
            else:
                mappings.append((col, col, 0))  # Keep original name
        
        return mappings
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate data quality score based on completeness and validity
        """
        # Missing data penalty
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        completeness_score = 1 - missing_pct
        
        # Numeric columns should have reasonable values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        validity_score = 1.0
        
        for col in numeric_cols:
            if df[col].min() < 0 and 'amt' in col.lower():
                # Negative amounts reduce validity
                validity_score *= 0.95
        
        # Combined score
        quality_score = (completeness_score * 0.7 + validity_score * 0.3)
        
        return quality_score
    
    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure minimum required columns exist"""
        # Account ID
        if 'account_id' not in df.columns:
            df['account_id'] = df.index.astype(str)
        
        # Credit Limit
        if 'credit_limit' not in df.columns:
            df['credit_limit'] = 50000  # Default if missing
        
        # Outstanding Balance - calculate from bill amounts or use 30% of credit limit
        if 'outstanding_balance' not in df.columns:
            # Try multiple columns that might represent outstanding balance
            balance_candidates = ['bill_amount_1', 'bill_amt1', 'balance', 'current_balance', 
                                'outstanding', 'loan_amnt', 'funded_amnt']
            
            balance_set = False
            for col in balance_candidates:
                if col in df.columns:
                    try:
                        # Ensure we're getting a Series, not a DataFrame
                        col_data = df[col]
                        if isinstance(col_data, pd.Series):
                            df['outstanding_balance'] = pd.to_numeric(col_data, errors='coerce').fillna(0)
                            balance_set = True
                            break
                    except:
                        continue
            
            # If no suitable column found, use 30% of credit limit as reasonable default
            if not balance_set:
                df['outstanding_balance'] = df['credit_limit'] * 0.3
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        # Convert numeric columns
        numeric_patterns = ['amount', 'amt', 'limit', 'balance', 'pay_']
        
        for col in df.columns:
            if any(pattern in col.lower() for pattern in numeric_patterns):
                try:
                    # Ensure we're working with a Series
                    col_data = df[col]
                    if isinstance(col_data, pd.Series):
                        df[col] = pd.to_numeric(col_data, errors='coerce').fillna(0)
                except Exception as e:
                    # If conversion fails, leave column as-is
                    print(f"Warning: Could not convert column '{col}' to numeric: {e}")
                    continue
        
        # Handle missing values for demographic columns
        try:
            if 'age' in df.columns:
                age_median = df['age'].median() if pd.api.types.is_numeric_dtype(df['age']) else 35
                df['age'] = df['age'].fillna(age_median)
            
            if 'sex' in df.columns:
                sex_mode = df['sex'].mode()[0] if len(df['sex'].mode()) > 0 else 1
                df['sex'] = df['sex'].fillna(sex_mode)
            
            if 'education' in df.columns:
                edu_mode = df['education'].mode()[0] if len(df['education'].mode()) > 0 else 2
                df['education'] = df['education'].fillna(edu_mode)
            
            if 'marriage' in df.columns:
                mar_mode = df['marriage'].mode()[0] if len(df['marriage'].mode()) > 0 else 1
                df['marriage'] = df['marriage'].fillna(mar_mode)
        except Exception as e:
            # If demographic filling fails, it's not critical
            print(f"Warning: Could not fill demographic columns: {e}")
        
        return df
