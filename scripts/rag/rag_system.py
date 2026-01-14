"""
RAG System for Insurance Customer Analytics
Uses natural language to query customer predictions (SQL or Pandas fallback)
"""

import mysql.connector
import pandas as pd
import re
import logging
from typing import Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class InsuranceRAGSystem:
    """RAG system that works with both SQL and Pandas fallback."""
    
    def __init__(self, df: pd.DataFrame = None, host='localhost', user='root', password='', database='insurance'):
        """Initialize with either a DataFrame or DB config."""
        self.df = df
        self.db_config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database
        }
        
    def _get_connection(self):
        """Create database connection"""
        return mysql.connector.connect(**self.db_config)
    
    def _parse_query(self, question: str) -> Dict[str, any]:
        """Parse natural language question into query parameters"""
        question_lower = question.lower()
        
        params = {
            'limit': 10,  # default
            'order_by': 'churn_probability DESC',
            'filters': [],
            'focus': 'general'
        }
        
        # Extract number from "top N" or "show N"
        top_match = re.search(r'\b(?:top|show|first|best|worst|display|list|find)\s+(\d+)\b', question_lower)
        if top_match:
            params['limit'] = int(top_match.group(1))
        
        # Determine focus and ordering
        if any(word in question_lower for word in ['churn', 'leave', 'retention', 'lapse', 'quit']):
            params['focus'] = 'churn'
            params['order_by'] = 'churn_probability DESC'
        elif any(word in question_lower for word in ['claim', 'accident', 'payout', 'incident']):
            params['focus'] = 'claims'
            params['order_by'] = 'claims_probability DESC'
        elif any(word in question_lower for word in ['value', 'clv', 'lifetime', 'worth', 'valuable', 'revenue']):
            params['focus'] = 'value'
            params['order_by'] = 'customer_lifetime_value DESC'
        elif any(word in question_lower for word in ['renewal', 'renew']):
            params['focus'] = 'renewal'
            params['order_by'] = 'renewal_risk_score DESC'
        
        # Filters: Risk Levels
        if 'critical' in question_lower or 'highest risk' in question_lower:
            params['filters'].append(("churn_probability >= 0.7", "Critical Risk"))
        elif 'warning' in question_lower or 'medium risk' in question_lower:
            params['filters'].append(("churn_probability BETWEEN 0.4 AND 0.7", "Medium Risk"))
        elif 'low risk' in question_lower or 'safe' in question_lower:
            params['filters'].append(("churn_probability < 0.4", "Low Risk"))
        
        # Filters: Value Levels
        if 'high value' in question_lower or 'vip' in question_lower:
            params['filters'].append(("customer_lifetime_value >= 1200", "High Value"))
        elif 'low value' in question_lower or 'small' in question_lower:
            params['filters'].append(("customer_lifetime_value < 600", "Low Value"))
        
        # Filters: Segments
        for segment in ['platinum', 'gold', 'silver', 'bronze']:
            if segment in question_lower:
                params['filters'].append((f"customer_segment = '{segment.capitalize()}'", f"{segment.capitalize()} segment"))
        
        # Filters: Journey Quadrants
        for quadrant in ['protect', 'grow', 'rescue', 'monitor']:
            if quadrant in question_lower:
                params['filters'].append((f"journey_quadrant = '{quadrant.capitalize()}'", f"{quadrant.capitalize()} quadrant"))
        
        # Filters: Special Flags
        if 'underpriced' in question_lower or 'low premium' in question_lower:
            params['filters'].append(("pricing_adequacy_flag = 1", "Underpriced Policies"))
            params['order_by'] = 'customer_lifetime_value ASC'
            
        return params

    def _query_pandas(self, params: Dict) -> pd.DataFrame:
        """Query the in-memory DataFrame using pandas filters."""
        if self.df is None:
            return pd.DataFrame()
        
        filtered_df = self.df.copy()
        
        # Column mapping for standardization
        col_map = {
            'policy_id': 'ID',
            'churn_probability': 'Churn_Prob',
            'claims_probability': 'Claims_Prob',
            'customer_lifetime_value': 'CLV',
            'customer_segment': 'Segment',
            'journey_quadrant': 'Journey',
            'pricing_adequacy_flag': 'Underpriced',
            'renewal_risk_score': 'Renewal_Risk'
        }
        
        # Apply filters
        for sql_filter, desc in params['filters']:
            if 'churn_probability' in sql_filter:
                col = col_map['churn_probability']
                if '>=' in sql_filter: filtered_df = filtered_df[filtered_df[col] >= 0.7]
                elif 'BETWEEN' in sql_filter: filtered_df = filtered_df[(filtered_df[col] >= 0.4) & (filtered_df[col] < 0.7)]
                elif '<' in sql_filter: filtered_df = filtered_df[filtered_df[col] < 0.4]
            
            elif 'customer_lifetime_value' in sql_filter:
                col = col_map['customer_lifetime_value']
                if '>=' in sql_filter: filtered_df = filtered_df[filtered_df[col] >= 1200]
                elif '<' in sql_filter: filtered_df = filtered_df[filtered_df[col] < 600]
                
            elif 'customer_segment' in sql_filter:
                val = re.search(r"'(.*?)'", sql_filter).group(1)
                filtered_df = filtered_df[filtered_df[col_map['customer_segment']] == val]
                
            elif 'journey_quadrant' in sql_filter:
                val = re.search(r"'(.*?)'", sql_filter).group(1)
                filtered_df = filtered_df[filtered_df[col_map['journey_quadrant']] == val]
            
            elif 'pricing_adequacy_flag' in sql_filter:
                filtered_df = filtered_df[filtered_df[col_map['pricing_adequacy_flag']] == 1]

        # Apply ordering
        order_col = params['order_by'].split(' ')[0]
        ascending = 'DESC' not in params['order_by']
        
        df_order_col = col_map.get(order_col, order_col)
        if df_order_col in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by=df_order_col, ascending=ascending)
        
        return filtered_df.head(params['limit'])

    def query(self, question: str) -> Tuple[pd.DataFrame, str]:
        """
        Main query interface
        Returns: (DataFrame of results, explanation text)
        """
        params = self._parse_query(question)
        
        # Check if we have an in-memory dataframe (preferred for Cloud/No-SQL)
        if self.df is not None and not self.df.empty:
            logger.info(f"RAG: Querying in-memory DataFrame ({len(self.df)} rows)")
            results_df = self._query_pandas(params)
        else:
            # Fallback to SQL if no DataFrame is provided
            logger.info("RAG: No DataFrame provided, attempting SQL query")
            try:
                # Build SQL query
                select_cols = "policy_id, churn_probability, claims_probability, customer_lifetime_value, customer_segment, journey_quadrant"
                query = f"SELECT {select_cols} FROM model_predictions"
                
                if params['filters']:
                    filter_sql = " AND ".join(f[0] for f in params['filters'])
                    query += f" WHERE {filter_sql}"
                
                query += f" ORDER BY {params['order_by']} LIMIT {params['limit']}"
                
                conn = self._get_connection()
                results_df = pd.read_sql(query, conn)
                conn.close()
                
                # Standardize results column names for consistency with pandas output
                results_df = results_df.rename(columns={
                    'policy_id': 'ID',
                    'churn_probability': 'Churn_Prob',
                    'claims_probability': 'Claims_Prob',
                    'customer_lifetime_value': 'CLV',
                    'customer_segment': 'Segment',
                    'journey_quadrant': 'Journey'
                })
            except Exception as e:
                logger.error(f"RAG: SQL query failed: {e}")
                results_df = pd.DataFrame()
        
        # Generate explanation
        explanation = self._generate_explanation(question, params, results_df)
        
        return results_df, explanation
    
    def _generate_explanation(self, question: str, params: Dict, df: pd.DataFrame) -> str:
        """Generate human-readable explanation of results"""
        
        if len(df) == 0:
            return "No customers match your criteria in the current portfolio data."
            
        parts = []
        
        # Intro
        if params['filters']:
            filter_desc = ", ".join(f[1] for f in params['filters'])
            parts.append(f"Found **{len(df)} customers** matching: {filter_desc}")
        else:
            parts.append(f"Listing top **{len(df)} customers** based on your request.")
        
        # Insights based on focus
        if params['focus'] == 'churn' and 'Churn_Prob' in df.columns:
            avg_churn = df['Churn_Prob'].mean()
            parts.append(f"Average churn probability: **{avg_churn:.1%}**")
            high_risk = (df['Churn_Prob'] >= 0.7).sum()
            if high_risk > 0:
                parts.append(f"⚠️ **{high_risk}** customers at critical churn risk (>=70%)")
        
        elif params['focus'] == 'claims' and 'Claims_Prob' in df.columns:
            avg_claims = df['Claims_Prob'].mean()
            parts.append(f"Average claims probability: **{avg_claims:.1%}**")
        
        elif params['focus'] == 'value' and 'CLV' in df.columns:
            total_value = df['CLV'].sum()
            parts.append(f"Total segment portfolio value: **€{total_value:,.0f}**")
            parts.append(f"Average CLV: **€{df['CLV'].mean():,.0f}**")
            
        return "\n\n".join(parts)

    def get_statistics(self) -> Dict:
        """Get overall portfolio statistics"""
        if self.df is not None:
            return {
                'total_customers': len(self.df),
                'avg_churn': float(self.df['Churn_Prob'].mean()),
                'avg_claims': float(self.df['Claims_Prob'].mean()),
                'total_value': float(self.df['CLV'].sum()),
                'avg_value': float(self.df['CLV'].mean())
            }
        
        try:
            conn = self._get_connection()
            total_query = """
            SELECT 
                COUNT(*) as total_customers,
                AVG(churn_probability) as avg_churn,
                AVG(claims_probability) as avg_claims,
                SUM(customer_lifetime_value) as total_value,
                AVG(customer_lifetime_value) as avg_value
            FROM model_predictions
            """
            totals = pd.read_sql(total_query, conn).iloc[0]
            conn.close()
            
            return {
                'total_customers': int(totals['total_customers']),
                'avg_churn': float(totals['avg_churn']),
                'avg_claims': float(totals['avg_claims']),
                'total_value': float(totals['total_value']),
                'avg_value': float(totals['avg_value'])
            }
        except Exception:
            return {}

# Example usage
if __name__ == "__main__":
    rag = InsuranceRAGSystem()
    test_questions = ["Show top 5 churn risk customers"]
    for q in test_questions:
        df, explanation = rag.query(q)
        print(explanation)
        print(df.head())

