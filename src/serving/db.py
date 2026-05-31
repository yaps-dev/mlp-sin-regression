import os
from sqlalchemy import create_engine

db_user = os.getenv("DB_USER", "username")
db_user_password = os.getenv("DB_PASSWORD", "password")
db_addr = os.getenv("DB_ADDR", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_schema = os.getenv("DB_SCHEMA", "ml_data")
# Read-only user!
engine = create_engine(
    f"postgresql://{db_user}:{db_user_password}@{db_addr}:{db_port}/{db_schema}",
    pool_size=2,
    pool_pre_ping=True,
    pool_recycle=3600,
)