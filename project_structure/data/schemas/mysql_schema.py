"""
Database schema for insurance platform.
Normalizes motor vehicle insurance data for operational use.
"""

SCHEMA_SQL = """
-- Main schema
CREATE SCHEMA IF NOT EXISTS insurance;

-- Customers table
CREATE TABLE insurance.customers (
  customer_id INT AUTO_INCREMENT PRIMARY KEY,
  date_birth DATE NOT NULL,
  date_driving_licence DATE NOT NULL,
  -- age_group: compute in ETL/application (keeps schema portable)
  age_group VARCHAR(10) DEFAULT NULL,
  -- licence_years: compute in ETL/application
  licence_years INT DEFAULT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_age_group (age_group),
  INDEX idx_licence_years (licence_years)
);

-- Vehicles table
CREATE TABLE insurance.vehicles (
  vehicle_id INT AUTO_INCREMENT PRIMARY KEY,
  year_matriculation INT NOT NULL,
  power INT,
  cylinder_capacity INT,
  value_vehicle DECIMAL(10, 2),
  n_doors INT,
  type_fuel ENUM('P', 'D', 'G', 'H', 'E', 'LP') NOT NULL,
  length DECIMAL(5, 2),
  weight INT,
  vehicle_age INT DEFAULT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_vehicle_age (vehicle_age),
  INDEX idx_type_fuel (type_fuel)
);

-- Policies table
CREATE TABLE insurance.policies (
  policy_id INT PRIMARY KEY,
  customer_id INT NOT NULL,
  vehicle_id INT NOT NULL,
  second_driver TINYINT DEFAULT 0,
  date_start_contract DATE NOT NULL,
  date_last_renewal DATE,
  date_next_renewal DATE,
  date_lapse DATE,
  lapse TINYINT DEFAULT 0,
  distribution_channel INT,
  seniority INT,
  policies_in_force INT,
  max_policies INT,
  max_products INT,
  payment INT DEFAULT 0,
  premium DECIMAL(10, 2) NOT NULL,
  cost_claims_year DECIMAL(10, 2) DEFAULT 0,
  n_claims_year INT DEFAULT 0,
  n_claims_history INT DEFAULT 0,
  r_claims_history DECIMAL(5, 2) DEFAULT 0,
  type_risk INT,
  area INT,
  contract_days INT DEFAULT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (customer_id) REFERENCES insurance.customers(customer_id) ON DELETE CASCADE,
  FOREIGN KEY (vehicle_id) REFERENCES insurance.vehicles(vehicle_id) ON DELETE CASCADE,
  INDEX idx_lapse (lapse),
  INDEX idx_start_contract (date_start_contract),
  INDEX idx_customer_id (customer_id),
  INDEX idx_vehicle_id (vehicle_id),
  INDEX idx_premium (premium)
);

-- Claims table
CREATE TABLE insurance.claims (
  claim_id INT AUTO_INCREMENT PRIMARY KEY,
  policy_id INT NOT NULL,
  claim_date DATE NOT NULL,
  claim_amount DECIMAL(10, 2) NOT NULL,
  claim_type VARCHAR(100),
  claim_status ENUM('open', 'approved', 'denied', 'settled') DEFAULT 'open',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (policy_id) REFERENCES insurance.policies(policy_id) ON DELETE CASCADE,
  INDEX idx_policy_id (policy_id),
  INDEX idx_claim_date (claim_date),
  INDEX idx_claim_status (claim_status)
);

-- Model predictions cache
CREATE TABLE insurance.predictions (
  prediction_id INT AUTO_INCREMENT PRIMARY KEY,
  policy_id INT NOT NULL,
  model_name VARCHAR(100) NOT NULL,
  prediction_type ENUM('lapse', 'claim_amount', 'risk_score') NOT NULL,
  predicted_value DECIMAL(10, 4),
  prediction_probability DECIMAL(5, 4),
  actual_value DECIMAL(10, 4),
  feature_version INT,
  model_version INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (policy_id) REFERENCES insurance.policies(policy_id) ON DELETE CASCADE,
  INDEX idx_policy_id (policy_id),
  INDEX idx_model_name (model_name),
  INDEX idx_prediction_type (prediction_type),
  INDEX idx_created_at (created_at)
);

-- Feature store for historical features
CREATE TABLE insurance.feature_store (
  feature_id INT AUTO_INCREMENT PRIMARY KEY,
  policy_id INT NOT NULL,
  feature_name VARCHAR(200) NOT NULL,
  feature_value VARCHAR(500),
  feature_timestamp TIMESTAMP NOT NULL,
  feature_version INT DEFAULT 1,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (policy_id) REFERENCES insurance.policies(policy_id) ON DELETE CASCADE,
  UNIQUE KEY unique_feature (policy_id, feature_name, feature_timestamp),
  INDEX idx_policy_id (policy_id),
  INDEX idx_feature_name (feature_name),
  INDEX idx_feature_timestamp (feature_timestamp)
);

-- Audit/Log table for tracking ETL and model changes
CREATE TABLE insurance.audit_log (
  log_id INT AUTO_INCREMENT PRIMARY KEY,
  entity_type VARCHAR(50) NOT NULL,
  entity_id INT,
  action VARCHAR(50) NOT NULL,
  changes JSON,
  created_by VARCHAR(100),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_entity_type (entity_type),
  INDEX idx_created_at (created_at)
);
"""
