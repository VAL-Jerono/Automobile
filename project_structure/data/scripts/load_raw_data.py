"""
Load raw CSV data into MySQL database.
"""

import pandas as pd
import mysql.connector
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def load_csv_to_mysql():
    """Read Motor_vehicle_insurance_data.csv and load into MySQL."""
    csv_path = 'data/raw/Motor_vehicle_insurance_data.csv'
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        return

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path, sep=';', low_memory=False, encoding='utf-8')
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Connect to MySQL
    conn = mysql.connector.connect(
        host=os.getenv('MYSQL_HOST', 'localhost'),
        user=os.getenv('MYSQL_USER', 'root'),
        password=os.getenv('MYSQL_PASSWORD', ''),
        port=int(os.getenv('MYSQL_PORT', 3306)),
        database=os.getenv('MYSQL_DATABASE', 'insurance_db')
    )
    cursor = conn.cursor()

    # Parse dates
    date_cols = ['Date_start_contract', 'Date_last_renewal', 'Date_next_renewal', 
                 'Date_birth', 'Date_driving_licence', 'Date_lapse']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    # Load customers (unique by DOB + driving license date)
    print("Loading customers...")
    customers_df = df[['Date_birth', 'Date_driving_licence']].drop_duplicates()
    for _, row in tqdm(customers_df.iterrows(), total=len(customers_df)):
        sql = "INSERT IGNORE INTO insurance.customers (date_birth, date_driving_licence) VALUES (%s, %s)"
        cursor.execute(sql, (row['Date_birth'], row['Date_driving_licence']))
    conn.commit()
    print(f"✓ Customers loaded")

    # Load vehicles (unique by year, power, etc.)
    print("Loading vehicles...")
    vehicles_df = df[['Year_matriculation', 'Power', 'Cylinder_capacity', 'Value_vehicle', 
                      'N_doors', 'Type_fuel', 'Length', 'Weight']].drop_duplicates()
    for _, row in tqdm(vehicles_df.iterrows(), total=len(vehicles_df)):
        sql = """INSERT IGNORE INTO insurance.vehicles 
                 (year_matriculation, power, cylinder_capacity, value_vehicle, n_doors, type_fuel, length, weight) 
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor.execute(sql, (row['Year_matriculation'], row['Power'], row['Cylinder_capacity'], 
                             row['Value_vehicle'], row['N_doors'], row['Type_fuel'], 
                             row['Length'], row['Weight']))
    conn.commit()
    print(f"✓ Vehicles loaded")

    # Load policies
    print("Loading policies...")
    # Pre-fetch all customer and vehicle mappings to avoid repeated SELECT queries
    cursor.execute("SELECT customer_id, date_birth, date_driving_licence FROM insurance.customers")
    cust_map = {}
    for cust_id, dob, lic_date in cursor.fetchall():
        key = (pd.Timestamp(dob), pd.Timestamp(lic_date))
        cust_map[key] = cust_id

    cursor.execute("SELECT vehicle_id, year_matriculation, power, type_fuel FROM insurance.vehicles")
    veh_map = {}
    for veh_id, year, power, fuel in cursor.fetchall():
        key = (year, power, fuel)
        veh_map[key] = veh_id

    for _, row in tqdm(df.iterrows(), total=len(df)):
        cust_key = (pd.Timestamp(row['Date_birth']), pd.Timestamp(row['Date_driving_licence']))
        customer_id = cust_map.get(cust_key)

        veh_key = (int(row['Year_matriculation']), row['Power'], row['Type_fuel'])
        vehicle_id = veh_map.get(veh_key)

        if customer_id and vehicle_id:
            # Convert NaT (NaN datetime) to None for MySQL NULL
            date_start = None if pd.isna(row['Date_start_contract']) else row['Date_start_contract']
            date_last_renewal = None if pd.isna(row['Date_last_renewal']) else row['Date_last_renewal']
            date_next_renewal = None if pd.isna(row['Date_next_renewal']) else row['Date_next_renewal']
            date_lapse = None if pd.isna(row['Date_lapse']) else row['Date_lapse']
            
            sql = """INSERT IGNORE INTO insurance.policies 
                     (policy_id, customer_id, vehicle_id, second_driver, date_start_contract, 
                      date_last_renewal, date_next_renewal, date_lapse, lapse, 
                      distribution_channel, seniority, policies_in_force, max_policies, max_products,
                      payment, premium, cost_claims_year, n_claims_year, n_claims_history, 
                      r_claims_history, type_risk, area)
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql, (
                int(row['ID']), customer_id, vehicle_id, int(row['Second_driver']),
                date_start, date_last_renewal, date_next_renewal,
                date_lapse, int(row['Lapse']), int(row['Distribution_channel']), int(row['Seniority']),
                int(row['Policies_in_force']), int(row['Max_policies']), int(row['Max_products']),
                int(row['Payment']), float(row['Premium']), float(row['Cost_claims_year']), int(row['N_claims_year']),
                int(row['N_claims_history']), float(row['R_Claims_history']), int(row['Type_risk']), int(row['Area'])
            ))
    conn.commit()
    print(f"✓ Policies loaded")

    cursor.close()
    conn.close()
    print("✓ Data loading complete!")

if __name__ == '__main__':
    load_csv_to_mysql()
