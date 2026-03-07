import json
import pandas as pd
from typing import Dict, Any, Tuple
from sqlalchemy import create_engine
from pymongo import MongoClient
import boto3
from google.cloud import storage as gcs
from azure.storage.blob import BlobServiceClient

class UniversalConnector:
    """
    Handles PULL and PUSH operations for Databases and Cloud Storage.
    Credentials are fully ephemeral and passed at runtime.
    """
    
    @staticmethod
    def test_database_connection(platform: str, credentials: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            if platform == "PostgreSQL":
                uri = credentials.get("uri") or f"postgresql://{credentials['username']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['database']}"
                engine = create_engine(uri)
                with engine.connect() as conn:
                    pass
                return True, "PostgreSQL connected successfully."
            elif platform == "MySQL":
                uri = credentials.get("uri") or f"mysql+pymysql://{credentials['username']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['database']}"
                engine = create_engine(uri)
                with engine.connect() as conn:
                    pass
                return True, "MySQL connected successfully."
            elif platform == "MongoDB":
                uri = credentials.get("connection_uri") or f"mongodb://{credentials['username']}:{credentials['password']}@{credentials['host']}:{credentials['port']}"
                client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                client.server_info()
                return True, "MongoDB connected successfully."
            # Snowflake and BigQuery skipped for local testing mock, but should follow SQLAlchemy
            return False, f"Platform {platform} testing not fully implemented."
        except Exception as e:
            return False, str(e)

    @staticmethod
    def pull_database(platform: str, credentials: Dict[str, Any], target_table: str) -> str:
        """Pulls data from a database and returns it as a JSON string (simulating a document load)."""
        if platform in ["PostgreSQL", "MySQL"]:
            uri = credentials.get("uri")
            if not uri:
                prefix = "postgresql" if platform == "PostgreSQL" else "mysql+pymysql"
                uri = f"{prefix}://{credentials['username']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['database']}"
            engine = create_engine(uri)
            df = pd.read_sql_table(target_table, engine)
            return df.to_json(orient="records")
        elif platform == "MongoDB":
            uri = credentials.get("connection_uri") or f"mongodb://{credentials['username']}:{credentials['password']}@{credentials['host']}:{credentials['port']}"
            client = MongoClient(uri)
            db = client[credentials.get("database", "default")]
            collection = db[target_table]
            data = list(collection.find({}, {"_id": 0}))
            return json.dumps(data)
        return "[]"

    @staticmethod
    def push_database(platform: str, credentials: Dict[str, Any], target_table: str, data: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            if platform in ["PostgreSQL", "MySQL"]:
                uri = credentials.get("uri")
                if not uri:
                    prefix = "postgresql" if platform == "PostgreSQL" else "mysql+pymysql"
                    uri = f"{prefix}://{credentials['username']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['database']}"
                engine = create_engine(uri)
                df.to_sql(target_table, engine, if_exists="append", index=False)
                return True, f"Pushed {len(df)} rows to {target_table} in {platform}."
            elif platform == "MongoDB":
                uri = credentials.get("connection_uri") or f"mongodb://{credentials['username']}:{credentials['password']}@{credentials['host']}:{credentials['port']}"
                client = MongoClient(uri)
                db = client[credentials.get("database", "default")]
                collection = db[target_table]
                records = df.to_dict(orient="records")
                if records:
                    collection.insert_many(records)
                return True, f"Pushed {len(records)} docs to {target_table} in MongoDB."
            return False, f"Push for {platform} not fully implemented."
        except Exception as e:
            return False, str(e)
