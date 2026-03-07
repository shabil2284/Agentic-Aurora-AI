import os
import json
import tempfile
from typing import Dict, Any, Tuple, List
import boto3
from google.cloud import storage as gcs
from azure.storage.blob import BlobServiceClient

class UniversalStorageConnector:
    """Handles PULL (list/download) and PUSH (upload) for Cloud Storage."""
    
    @staticmethod
    def list_files(platform: str, credentials: Dict[str, Any], container: str) -> List[str]:
        try:
            if platform == "AWS S3":
                s3 = boto3.client('s3', 
                    aws_access_key_id=credentials.get("access_key"),
                    aws_secret_access_key=credentials.get("secret_key"),
                    region_name=credentials.get("region", "us-east-1")
                )
                response = s3.list_objects_v2(Bucket=container)
                return [obj['Key'] for obj in response.get('Contents', [])]
            elif platform == "GCS":
                client = gcs.Client.from_service_account_info(json.loads(credentials.get("service_account_json", "{}")))
                bucket = client.bucket(container)
                return [blob.name for blob in bucket.list_blobs()]
            elif platform == "Azure Blob":
                service_client = BlobServiceClient.from_connection_string(credentials.get("connection_string"))
                container_client = service_client.get_container_client(container)
                return [blob.name for blob in container_client.list_blobs()]
            return []
        except Exception as e:
            print(f"Error listing files for {platform}: {e}")
            return []

    @staticmethod
    def pull_file(platform: str, credentials: Dict[str, Any], container: str, filename: str) -> str:
        """Downloads file to a temporary location and returns the path."""
        ext = os.path.splitext(filename)[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.close()
        try:
            if platform == "AWS S3":
                s3 = boto3.client('s3', 
                    aws_access_key_id=credentials.get("access_key"),
                    aws_secret_access_key=credentials.get("secret_key"),
                    region_name=credentials.get("region", "us-east-1")
                )
                s3.download_file(container, filename, tmp.name)
            elif platform == "GCS":
                client = gcs.Client.from_service_account_info(json.loads(credentials.get("service_account_json", "{}")))
                blob = client.bucket(container).blob(filename)
                blob.download_to_filename(tmp.name)
            elif platform == "Azure Blob":
                service_client = BlobServiceClient.from_connection_string(credentials.get("connection_string"))
                blob_client = service_client.get_blob_client(container=container, blob=filename)
                with open(tmp.name, "wb") as f:
                    f.write(blob_client.download_blob().readall())
            return tmp.name
        except Exception as e:
            print(f"Error pulling file {filename} from {platform}: {e}")
            os.remove(tmp.name)
            return ""

    @staticmethod
    def push_file(platform: str, credentials: Dict[str, Any], container: str, file_path: str, destination_name: str) -> Tuple[bool, str]:
        try:
            if platform == "AWS S3":
                s3 = boto3.client('s3', 
                    aws_access_key_id=credentials.get("access_key"),
                    aws_secret_access_key=credentials.get("secret_key"),
                    region_name=credentials.get("region", "us-east-1")
                )
                s3.upload_file(file_path, container, destination_name)
                return True, f"Uploaded to s3://{container}/{destination_name}"
            elif platform == "GCS":
                client = gcs.Client.from_service_account_info(json.loads(credentials.get("service_account_json", "{}")))
                blob = client.bucket(container).blob(destination_name)
                blob.upload_from_filename(file_path)
                return True, f"Uploaded to gs://{container}/{destination_name}"
            elif platform == "Azure Blob":
                service_client = BlobServiceClient.from_connection_string(credentials.get("connection_string"))
                blob_client = service_client.get_blob_client(container=container, blob=destination_name)
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                return True, f"Uploaded to Azure container {container} as {destination_name}"
            return False, f"Push for {platform} not fully implemented."
        except Exception as e:
            return False, str(e)

    @staticmethod
    def verify_push(
        platform: str,
        credentials: Dict[str, Any],
        container: str,
        destination_name: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify that a pushed object exists remotely and has size > 0.
        Returns (verified, {"size": int, "exists": bool, "error": str|None}).
        """
        result: Dict[str, Any] = {"exists": False, "size": -1, "error": None}
        try:
            if platform == "AWS S3":
                s3 = boto3.client(
                    "s3",
                    aws_access_key_id=credentials.get("access_key"),
                    aws_secret_access_key=credentials.get("secret_key"),
                    region_name=credentials.get("region", "us-east-1"),
                )
                resp = s3.head_object(Bucket=container, Key=destination_name)
                result["exists"] = True
                result["size"] = resp.get("ContentLength", 0)

            elif platform == "GCS":
                client = gcs.Client.from_service_account_info(
                    json.loads(credentials.get("service_account_json", "{}"))
                )
                blob = client.bucket(container).blob(destination_name)
                blob.reload()
                result["exists"] = blob.exists()
                result["size"] = blob.size or 0

            elif platform == "Azure Blob":
                service_client = BlobServiceClient.from_connection_string(
                    credentials.get("connection_string")
                )
                blob_client = service_client.get_blob_client(
                    container=container, blob=destination_name
                )
                props = blob_client.get_blob_properties()
                result["exists"] = True
                result["size"] = props.size or 0

            verified = result["exists"] and result["size"] > 0
            return verified, result

        except Exception as e:
            result["error"] = str(e)
            return False, result

