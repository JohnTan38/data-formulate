# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import duckdb
import pandas as pd
from typing import Dict
import tempfile
import os
from contextlib import contextmanager
from dotenv import load_dotenv
import psycopg2

class DuckDBManager:
    def __init__(self, local_db_dir: str):
        # Store session db file paths
        self._db_files: Dict[str, str] = {}
        self._local_db_dir: str = local_db_dir

    @contextmanager
    def connection(self, session_id: str):
        """Get a DuckDB connection as a context manager that will be closed when exiting the context"""
        conn = None
        try:
            conn = self.get_connection(session_id)
            yield conn
        finally:
            if conn:
                conn.close()
    
    def get_connection(self, session_id: str) -> duckdb.DuckDBPyConnection:
        """Internal method to get or create a DuckDB connection for a session"""
        # Get or create the db file path for this session
        if session_id not in self._db_files or self._db_files[session_id] is None:
            db_dir = self._local_db_dir if self._local_db_dir else tempfile.gettempdir()
            if not os.path.exists(db_dir):
                db_dir = tempfile.gettempdir()
            db_file = os.path.join(db_dir, f"df_{session_id}.duckdb")
            print(f"=== Creating new db file: {db_file}")
            self._db_files[session_id] = db_file
        else:
            print(f"=== Using existing db file: {self._db_files[session_id]}")
            db_file = self._db_files[session_id]
            
        # Create a fresh connection to the database file
        conn = duckdb.connect(database=db_file)

        return conn


# ...existing code...

def get_postgres_connection():
    """
    Return a psycopg2 connection using env vars.
    - If DB_HOST starts with '/cloudsql/' we use the unix socket path (Cloud Run + Cloud SQL).
    - Otherwise we connect via TCP host:port.
    Environment variables: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_DATABASE, DB_SSLMODE (optional)
    """
    import logging
    DB_HOST = os.getenv("DB_HOST", "")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_DATABASE = os.getenv("DB_DATABASE", "postgres")
    DB_SSLMODE = os.getenv("DB_SSLMODE", "")  # e.g. 'require' if needed

    try:
        if DB_HOST and DB_HOST.startswith("/cloudsql/"):
            # Connect over unix socket mounted by Cloud Run when --add-cloudsql-instances is used
            conn = psycopg2.connect(dbname=DB_DATABASE, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)
        elif DB_HOST:
            conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_DATABASE, user=DB_USER, password=DB_PASSWORD, sslmode=DB_SSLMODE or "prefer")
        else:
            raise RuntimeError("DB_HOST not set; cannot connect to Postgres")

        return conn
    except Exception as e:
        logging.exception("Unable to establish Postgres connection")
        raise
# ...existing code...

env = load_dotenv()

# Initialize the DB manager
db_manager = DuckDBManager(
    local_db_dir=os.getenv('LOCAL_DB_DIR')
)