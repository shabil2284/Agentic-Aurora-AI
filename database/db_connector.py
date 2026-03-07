from sqlalchemy import create_engine
import pandas as pd

class DBConnector:
    @staticmethod
    def get_engine(db_type: str, credentials: dict):
        if db_type == "PostgreSQL":
            url = f"postgresql://{credentials['user']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['database']}"
        elif db_type == "MySQL":
            url = f"mysql+pymysql://{credentials['user']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['database']}"
        elif db_type == "SQLite":
            url = f"sqlite:///{credentials['path']}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        return create_engine(url)

    @staticmethod
    def test_connection(engine) -> bool:
        try:
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

class MongoDBConnector:
    @staticmethod
    def get_client(uri: str):
        from pymongo import MongoClient
        return MongoClient(uri)
