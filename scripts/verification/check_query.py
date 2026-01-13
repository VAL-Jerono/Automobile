import mysql.connector

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='insurance'
)

cursor = conn.cursor()

# Check low risk + high value customers
cursor.execute('''
SELECT COUNT(*) as count, 
       MIN(churn_probability) as min_churn,
       MAX(churn_probability) as max_churn,
       MIN(customer_lifetime_value) as min_clv,
       MAX(customer_lifetime_value) as max_clv
FROM model_predictions
WHERE churn_probability < 0.4 AND customer_lifetime_value >= 1200
''')

result = cursor.fetchone()
print(f'Low risk (churn < 0.4) + High value (CLV >= 1200):')
print(f'  Count: {result[0]}')
if result[0] > 0:
    print(f'  Churn range: {result[1]:.3f} - {result[2]:.3f}')
    print(f'  CLV range: €{result[3]:.0f} - €{result[4]:.0f}')

conn.close()
