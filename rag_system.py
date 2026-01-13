"""
RAG System for Insurance Customer Analytics
Uses database queries to answer natural language questions about customers
"""

import mysql.connector
import pandas as pd
import re
from typing import Dict, List, Tuple

class InsuranceRAGSystem:
    """Simple RAG system using SQL queries on customer predictions"""
    
    def __init__(self, host='localhost', user='root', password='', database='insurance'):
        """Initialize connection to database"""
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
            'filters': []
        }
        
        # Extract number from "top N" or "show N"
        top_match = re.search(r'\b(?:top|show|first|best|worst)\s+(\d+)\b', question_lower)
        if top_match:
            params['limit'] = int(top_match.group(1))
        
        # Determine what to order by
        if any(word in question_lower for word in ['churn', 'leave', 'retention', 'lapse']):
            params['focus'] = 'churn'
            params['order_by'] = 'churn_probability DESC'
        elif any(word in question_lower for word in ['claim', 'accident']):
            params['focus'] = 'claims'
            params['order_by'] = 'claims_probability DESC'
        elif any(word in question_lower for word in ['value', 'clv', 'lifetime', 'worth', 'valuable']):
            params['focus'] = 'value'
            params['order_by'] = 'customer_lifetime_value DESC'
        elif any(word in question_lower for word in ['risk', 'critical', 'danger']):
            params['focus'] = 'risk'
            params['order_by'] = 'churn_probability DESC, customer_lifetime_value DESC'
        else:
            params['focus'] = 'general'
        
        # Look for risk level filters (churn-based only)
        if 'critical' in question_lower or 'high risk' in question_lower:
            params['filters'].append(("churn_probability >= 0.7", "Critical Risk"))
        elif 'warning' in question_lower or 'medium risk' in question_lower:
            params['filters'].append(("churn_probability BETWEEN 0.4 AND 0.7", "Medium Risk"))
        elif 'low risk' in question_lower or 'safe' in question_lower:
            params['filters'].append(("churn_probability < 0.4", "Low Risk"))
        
        # Look for value filters
        if 'high value' in question_lower or 'high clv' in question_lower:
            params['filters'].append(("customer_lifetime_value >= 1200", "High Value"))
        elif 'low value' in question_lower or 'low clv' in question_lower:
            params['filters'].append(("customer_lifetime_value < 600", "Low Value"))
        
        # Look for segment filters
        for segment in ['platinum', 'gold', 'silver', 'bronze']:
            if segment in question_lower:
                params['filters'].append((f"customer_segment = '{segment.capitalize()}'", f"{segment.capitalize()} segment"))
        
        # Look for quadrant filters
        for quadrant in ['protect', 'grow', 'rescue', 'monitor']:
            if quadrant in question_lower:
                params['filters'].append((f"journey_quadrant = '{quadrant.capitalize()}'", f"{quadrant.capitalize()} quadrant"))
        
        return params
    
    def query(self, question: str) -> Tuple[pd.DataFrame, str]:
        """
        Query the database based on natural language question
        Returns: (DataFrame of results, explanation text)
        """
        params = self._parse_query(question)
        
        # Build SQL query
        select_cols = """
            policy_id,
            churn_probability,
            claims_probability,
            customer_lifetime_value,
            customer_segment,
            journey_quadrant
        """
        
        query = f"SELECT {select_cols} FROM model_predictions"
        
        # Add filters
        if params['filters']:
            filter_sql = " AND ".join(f[0] for f in params['filters'])
            query += f" WHERE {filter_sql}"
        
        # Add ordering
        query += f" ORDER BY {params['order_by']}"
        
        # Add limit
        query += f" LIMIT {params['limit']}"
        
        # Execute query
        conn = self._get_connection()
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Generate explanation
        explanation = self._generate_explanation(question, params, df)
        
        return df, explanation
    
    def _generate_explanation(self, question: str, params: Dict, df: pd.DataFrame) -> str:
        """Generate human-readable explanation of results"""
        
        parts = []
        
        # Intro
        if params['filters']:
            filter_desc = ", ".join(f[1] for f in params['filters'])
            parts.append(f"Found **{len(df)} customers** matching: {filter_desc}")
        else:
            parts.append(f"Found **{len(df)} customers** from the portfolio")
        
        if len(df) == 0:
            return "No customers match your criteria."
        
        # Add focus-specific insights
        if params['focus'] == 'churn':
            avg_churn = df['churn_probability'].mean()
            parts.append(f"Average churn probability: **{avg_churn:.1%}**")
            high_risk = (df['churn_probability'] >= 0.7).sum()
            if high_risk > 0:
                parts.append(f"⚠️ **{high_risk}** customers at critical churn risk (≥70%)")
        
        elif params['focus'] == 'claims':
            avg_claims = df['claims_probability'].mean()
            parts.append(f"Average claims probability: **{avg_claims:.1%}**")
        
        elif params['focus'] == 'value':
            total_value = df['customer_lifetime_value'].sum()
            avg_value = df['customer_lifetime_value'].mean()
            parts.append(f"Total value: **€{total_value:,.0f}**")
            parts.append(f"Average CLV: **€{avg_value:,.0f}**")
        
        elif params['focus'] == 'risk':
            critical = (df['churn_probability'] >= 0.7) & (df['customer_lifetime_value'] >= 1200)
            if critical.any():
                critical_count = critical.sum()
                critical_value = df[critical]['customer_lifetime_value'].sum()
                parts.append(f"⚠️ **{critical_count}** critical risk customers")
                parts.append(f"At-risk value: **€{critical_value:,.0f}**")
        
        # Segment breakdown if diverse
        if df['customer_segment'].nunique() > 1:
            segments = df['customer_segment'].value_counts()
            seg_str = ", ".join([f"{seg}: {count}" for seg, count in segments.head(3).items()])
            parts.append(f"Segments: {seg_str}")
        
        return "\n\n".join(parts)
    
    def get_statistics(self) -> Dict:
        """Get overall portfolio statistics"""
        conn = self._get_connection()
        
        query = """
        SELECT 
            COUNT(*) as total_customers,
            AVG(churn_probability) as avg_churn,
            AVG(claims_probability) as avg_claims,
            SUM(customer_lifetime_value) as total_value,
            AVG(customer_lifetime_value) as avg_value,
            customer_segment,
            COUNT(*) as segment_count
        FROM model_predictions
        GROUP BY customer_segment
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Get totals
        total_query = """
        SELECT 
            COUNT(*) as total_customers,
            AVG(churn_probability) as avg_churn,
            AVG(claims_probability) as avg_claims,
            SUM(customer_lifetime_value) as total_value,
            AVG(customer_lifetime_value) as avg_value
        FROM model_predictions
        """
        
        conn = self._get_connection()
        totals = pd.read_sql(total_query, conn).iloc[0]
        conn.close()
        
        return {
            'total_customers': int(totals['total_customers']),
            'avg_churn': float(totals['avg_churn']),
            'avg_claims': float(totals['avg_claims']),
            'total_value': float(totals['total_value']),
            'avg_value': float(totals['avg_value']),
            'segments': df.to_dict('records')
        }

# Example usage
if __name__ == "__main__":
    rag = InsuranceRAGSystem()
    
    # Test queries
    test_questions = [
        "Show top 5 customers with highest churn risk",
        "Find 10 high value customers in critical risk",
        "Show customers in platinum segment with high churn",
        "List top 3 customers likely to make claims",
    ]
    
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print('='*60)
        df, explanation = rag.query(q)
        print(explanation)
        print(f"\nResults:\n{df.head()}")
