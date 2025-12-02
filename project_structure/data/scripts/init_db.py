"""
Initialize MySQL database and create schema.
Run this once before loading data.
"""

import mysql.connector
import os
from dotenv import load_dotenv
from data.schemas.mysql_schema import SCHEMA_SQL

load_dotenv()

def init_db():
    """Create database and schema."""
    try:
        # Connect to MySQL without selecting a database
        conn = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST', 'localhost'),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', ''),
            port=int(os.getenv('MYSQL_PORT', 3306))
        )
        cursor = conn.cursor()

        # Create database
        db_name = os.getenv('MYSQL_DATABASE', 'insurance_db')
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name};")
        cursor.execute(f"USE {db_name};")

        # Execute schema
        for statement in SCHEMA_SQL.split(';'):
            if statement.strip():
                cursor.execute(statement + ';')

        conn.commit()
        print(f"✓ Database '{db_name}' and schema created successfully.")
        cursor.close()
        conn.close()

    except mysql.connector.Error as err:
        print(f"✗ Database error: {err}")
        raise

if __name__ == '__main__':
    init_db()
