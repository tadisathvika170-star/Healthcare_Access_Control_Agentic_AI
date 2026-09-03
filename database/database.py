import sqlite3
from pathlib import Path


class DatabaseManager:
    """
    Handles all SQLite database operations.
    """

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.connection = None

    def connect(self):
        """Create a database connection."""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
        return self.connection

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute(self, query, params=()):
        """
        Execute INSERT, UPDATE, DELETE queries.
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()

    def fetchone(self, query, params=()):
        """
        Fetch one record.
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchone()

    def fetchall(self, query, params=()):
        """
        Fetch all records.
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()