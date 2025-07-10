import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="stock_analysis",
    user="postgres",
    password="user"  # Use your password here
)
print("Connection successful!")
conn.close()