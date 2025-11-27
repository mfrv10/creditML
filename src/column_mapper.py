"""
Manual Column Mapping UI Component
Provides an interactive Streamlit interface for users to manually map
document columns to expected credit portfolio fields.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


class ColumnMapper:
    """Interactive column mapping for credit portfolio data."""

    # Standard expected fields with descriptions
    FIELD_DEFINITIONS = {
        'account_id': 'Account or Customer ID',
        'credit_limit': 'Maximum Credit Limit',
        'balance': 'Current Outstanding Balance',
        'payment_status': 'Payment Status (0=current, 1-6=months late)',
        'age': 'Customer Age',
        'gender': 'Gender (1=Male, 2=Female)',
        'education': 'Education Level (1=grad school, 2=university, 3=high school, 4=others)',
        'marital_status': 'Marital Status (1=married, 2=single, 3=others)',
        'payment_history_1': 'Payment Status - Most Recent Month',
        'payment_history_2': 'Payment Status - 2 Months Ago',
        'payment_history_3': 'Payment Status - 3 Months Ago',
        'payment_history_4': 'Payment Status - 4 Months Ago',
        'payment_history_5': 'Payment Status - 5 Months Ago',
        'payment_history_6': 'Payment Status - 6 Months Ago',
        'bill_amount_1': 'Bill Amount - Most Recent Month',
        'bill_amount_2': 'Bill Amount - 2 Months Ago',
        'bill_amount_3': 'Bill Amount - 3 Months Ago',
        'bill_amount_4': 'Bill Amount - 4 Months Ago',
        'bill_amount_5': 'Bill Amount - 5 Months Ago',
        'bill_amount_6': 'Bill Amount - 6 Months Ago',
        'payment_amount_1': 'Payment Amount - Most Recent Month',
        'payment_amount_2': 'Payment Amount - 2 Months Ago',
        'payment_amount_3': 'Payment Amount - 3 Months Ago',
        'payment_amount_4': 'Payment Amount - 4 Months Ago',
        'payment_amount_5': 'Payment Amount - 5 Months Ago',
        'payment_amount_6': 'Payment Amount - 6 Months Ago',
        'default_status': 'Default Flag (0=no default, 1=default)',
    }

    # Required fields (minimum needed for analysis)
    REQUIRED_FIELDS = ['account_id', 'credit_limit', 'balance']

    def __init__(self):
        """Initialize column mapper."""
        self.mapping_cache_file = Path('.column_mappings.json')

    def render_mapping_ui(
        self,
        df: pd.DataFrame,
        auto_mapping: Optional[Dict[str, str]] = None
    ) -> Tuple[Dict[str, str], bool]:
        """
        Render interactive column mapping UI.

        Args:
            df: DataFrame with columns to map
            auto_mapping: Optional pre-computed automatic mapping

        Returns:
            Tuple of (mapping dict, ready flag)
        """
        st.subheader("🧩 Column Mapping Puzzle")
        st.write("Map your document columns to the expected fields for analysis.")

        # Initialize session state for mappings
        if 'column_mapping' not in st.session_state:
            st.session_state.column_mapping = auto_mapping or {}

        # Get detected columns
        detected_columns = list(df.columns)

        # Show mapping interface
        st.write("### Column Mappings")

        # Create two-column layout
        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**Your Columns**")
            st.caption(f"{len(detected_columns)} columns detected")

        with col2:
            st.write("**Expected Fields**")
            st.caption(f"{len(self.FIELD_DEFINITIONS)} standard fields")

        st.markdown("---")

        # Mapping interface
        mapping = {}
        mapped_source_cols = set()

        # Show required fields first
        st.write("#### ⚠️ Required Fields")
        for field in self.REQUIRED_FIELDS:
            mapping[field] = self._render_field_mapper(
                field,
                detected_columns,
                st.session_state.column_mapping.get(field),
                required=True
            )
            if mapping[field] and mapping[field] != '(Skip)':
                mapped_source_cols.add(mapping[field])

        st.markdown("---")

        # Show optional fields grouped by category
        st.write("#### Optional Fields")

        # Demographics
        with st.expander("👤 Customer Demographics", expanded=False):
            demo_fields = ['age', 'gender', 'education', 'marital_status']
            for field in demo_fields:
                mapping[field] = self._render_field_mapper(
                    field,
                    detected_columns,
                    st.session_state.column_mapping.get(field),
                    required=False
                )
                if mapping[field] and mapping[field] != '(Skip)':
                    mapped_source_cols.add(mapping[field])

        # Payment History
        with st.expander("📊 Payment History (6 months)", expanded=False):
            payment_fields = [f'payment_history_{i}' for i in range(1, 7)]
            cols = st.columns(2)
            for i, field in enumerate(payment_fields):
                with cols[i % 2]:
                    mapping[field] = self._render_field_mapper(
                        field,
                        detected_columns,
                        st.session_state.column_mapping.get(field),
                        required=False,
                        compact=True
                    )
                    if mapping[field] and mapping[field] != '(Skip)':
                        mapped_source_cols.add(mapping[field])

        # Bill Amounts
        with st.expander("💵 Bill Amounts (6 months)", expanded=False):
            bill_fields = [f'bill_amount_{i}' for i in range(1, 7)]
            cols = st.columns(2)
            for i, field in enumerate(bill_fields):
                with cols[i % 2]:
                    mapping[field] = self._render_field_mapper(
                        field,
                        detected_columns,
                        st.session_state.column_mapping.get(field),
                        required=False,
                        compact=True
                    )
                    if mapping[field] and mapping[field] != '(Skip)':
                        mapped_source_cols.add(mapping[field])

        # Payment Amounts
        with st.expander("💳 Payment Amounts (6 months)", expanded=False):
            payment_amt_fields = [f'payment_amount_{i}' for i in range(1, 7)]
            cols = st.columns(2)
            for i, field in enumerate(payment_amt_fields):
                with cols[i % 2]:
                    mapping[field] = self._render_field_mapper(
                        field,
                        detected_columns,
                        st.session_state.column_mapping.get(field),
                        required=False,
                        compact=True
                    )
                    if mapping[field] and mapping[field] != '(Skip)':
                        mapped_source_cols.add(mapping[field])

        # Other Fields
        with st.expander("🔧 Other Fields", expanded=False):
            other_fields = ['payment_status', 'default_status']
            for field in other_fields:
                mapping[field] = self._render_field_mapper(
                    field,
                    detected_columns,
                    st.session_state.column_mapping.get(field),
                    required=False
                )
                if mapping[field] and mapping[field] != '(Skip)':
                    mapped_source_cols.add(mapping[field])

        # Show unmapped columns
        unmapped_cols = set(detected_columns) - mapped_source_cols
        if unmapped_cols:
            st.markdown("---")
            st.write("#### ⚠️ Unmapped Columns")
            st.caption(f"{len(unmapped_cols)} columns will be ignored:")
            st.write(", ".join(sorted(unmapped_cols)))

        # Update session state
        st.session_state.column_mapping = mapping

        # Validate mapping
        required_mapped = all(
            mapping.get(field) and mapping[field] != '(Skip)'
            for field in self.REQUIRED_FIELDS
        )

        # Show status
        st.markdown("---")
        if required_mapped:
            st.success("✅ All required fields are mapped! Ready to proceed.")

            # Save mapping button
            if st.button("💾 Save This Mapping for Future Use"):
                self._save_mapping(mapping)
                st.success("Mapping saved!")
        else:
            missing = [f for f in self.REQUIRED_FIELDS
                      if not mapping.get(f) or mapping[f] == '(Skip)']
            st.error(f"⚠️ Missing required fields: {', '.join(missing)}")

        return mapping, required_mapped

    def _render_field_mapper(
        self,
        field_name: str,
        available_columns: List[str],
        default_value: Optional[str] = None,
        required: bool = False,
        compact: bool = False
    ) -> str:
        """
        Render a single field mapping selector.

        Args:
            field_name: Name of the target field
            available_columns: List of source columns to choose from
            default_value: Pre-selected value
            required: Whether this field is required
            compact: Use compact layout

        Returns:
            Selected source column or '(Skip)'
        """
        label = self.FIELD_DEFINITIONS.get(field_name, field_name)
        if required:
            label = f"🔴 {label} *"
        else:
            label = f"🟢 {label}"

        # Add "(Skip)" option for optional fields
        options = ['(Skip)'] + available_columns

        # Find default index
        if default_value and default_value in options:
            default_idx = options.index(default_value)
        else:
            default_idx = 0

        if compact:
            selected = st.selectbox(
                label,
                options,
                index=default_idx,
                key=f"map_{field_name}",
                label_visibility="visible"
            )
        else:
            col1, col2 = st.columns([2, 3])
            with col1:
                st.write(label)
                if not compact and field_name in self.FIELD_DEFINITIONS:
                    st.caption(self.FIELD_DEFINITIONS[field_name])
            with col2:
                selected = st.selectbox(
                    "Source column",
                    options,
                    index=default_idx,
                    key=f"map_{field_name}",
                    label_visibility="collapsed"
                )

        return selected if selected != '(Skip)' else None

    def apply_mapping(
        self,
        df: pd.DataFrame,
        mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Apply column mapping to transform DataFrame.

        Args:
            df: Source DataFrame
            mapping: Column mapping dict (target -> source)

        Returns:
            Transformed DataFrame with standardized columns
        """
        # Create new DataFrame with mapped columns
        result = pd.DataFrame()

        for target_field, source_field in mapping.items():
            if source_field and source_field in df.columns:
                result[target_field] = df[source_field]
            else:
                result[target_field] = None

        return result

    def _save_mapping(self, mapping: Dict[str, str]):
        """Save mapping to disk for future use."""
        try:
            with open(self.mapping_cache_file, 'w') as f:
                json.dump(mapping, f, indent=2)
        except Exception as e:
            st.warning(f"Could not save mapping: {e}")

    def load_saved_mapping(self) -> Optional[Dict[str, str]]:
        """Load previously saved mapping."""
        try:
            if self.mapping_cache_file.exists():
                with open(self.mapping_cache_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def show_mapping_summary(self, mapping: Dict[str, str], df: pd.DataFrame):
        """Show a summary of the current mapping."""
        st.write("### 📋 Mapping Summary")

        mapped_count = sum(1 for v in mapping.values() if v)
        total_fields = len(self.FIELD_DEFINITIONS)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mapped Fields", mapped_count)
        with col2:
            st.metric("Total Fields", total_fields)
        with col3:
            coverage = (mapped_count / total_fields) * 100
            st.metric("Coverage", f"{coverage:.0f}%")

        # Show mapping table
        mapping_data = []
        for target, source in mapping.items():
            if source:
                mapping_data.append({
                    'Target Field': target,
                    'Source Column': source,
                    'Sample Value': str(df[source].iloc[0]) if source in df.columns else 'N/A'
                })

        if mapping_data:
            st.dataframe(
                pd.DataFrame(mapping_data),
                use_container_width=True,
                hide_index=True
            )
