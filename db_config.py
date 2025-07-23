from dataclasses import dataclass

@dataclass
class DBConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "stock_analysis"
    user: str = "postgres"
    password: str = "password"  # Change this or use environment variables 
