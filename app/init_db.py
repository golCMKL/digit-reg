import os

import psycopg

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "digit_db"),
    "user": os.getenv("DB_USER", "digit_user"),
    "password": os.getenv("DB_PASSWORD", "digit_password"),
}


def init_db():
    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            # Create the table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    request_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    prediction INTEGER,
                    confidence REAL,
                    cached BOOLEAN NOT NULL,
                    latency_ms REAL,
                    error TEXT,
                    input_mean REAL,
                    input_std REAL,
                    input_min REAL,
                    input_max REAL,
                    nonzero_ratio REAL
                )
            """)

        conn.commit()

    print("Database initialized successfully")


if __name__ == "__main__":
    init_db()