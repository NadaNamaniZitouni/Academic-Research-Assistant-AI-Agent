"""
Database migration utilities
"""
from sqlalchemy import text
from .database import engine, SessionLocal
from .models import Base


def add_user_id_to_documents():
    """Add user_id column to documents table if it doesn't exist"""
    try:
        # Use raw connection to ensure it works
        conn = engine.raw_connection()
        cursor = conn.cursor()
        
        # Check if column exists
        cursor.execute("PRAGMA table_info(documents)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'user_id' not in columns:
            print("Adding user_id column to documents table...")
            cursor.execute("ALTER TABLE documents ADD COLUMN user_id TEXT")
            conn.commit()
            print("Successfully added user_id column to documents table")
        else:
            print("user_id column already exists in documents table")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error in migration: {e}")
        import traceback
        print(traceback.format_exc())
        try:
            conn.close()
        except:
            pass


def migrate_database():
    """Run all migrations"""
    print("Running database migrations...")
    add_user_id_to_documents()
    print("Database migrations completed.")


if __name__ == "__main__":
    migrate_database()

