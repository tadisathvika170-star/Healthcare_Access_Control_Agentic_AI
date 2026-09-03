"""
init_db.py

Initializes the users and audit log databases.
"""

from config import DATABASE_PATH, AUDIT_DATABASE_PATH
from database.database import DatabaseManager


def initialize_users_database():
    """
    Creates the users database.
    """

    db = DatabaseManager(DATABASE_PATH)

    create_users_table = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        full_name TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    db.execute(create_users_table)
    db.close()

    print("✅ users.db initialized successfully.")


def initialize_audit_database():
    """
    Creates the audit logs database.
    """

    db = DatabaseManager(AUDIT_DATABASE_PATH)

    create_audit_logs_table = """
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        action TEXT,
        status TEXT,
        confidence REAL,
        risk_level TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    db.execute(create_audit_logs_table)
    db.close()

    print("✅ audit_logs.db initialized successfully.")


def initialize_database():
    """
    Initializes all project databases.
    """

    initialize_users_database()
    initialize_audit_database()

    print("\n🎉 All databases initialized successfully.")


if __name__ == "__main__":
    initialize_database()