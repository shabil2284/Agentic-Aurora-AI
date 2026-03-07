import boto3
import os

class AWSConnector:
    def __init__(self, credentials: dict):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=credentials['access_key'],
            aws_secret_access_key=credentials['secret_key'],
            region_name=credentials.get('region', 'us-east-1'),
            aws_session_token=credentials.get('session_token')
        )

    def upload_file(self, local_path: str, bucket: str, key: str):
        try:
            self.s3.upload_file(local_path, bucket, key)
            return True, "Upload successful"
        except Exception as e:
            return False, str(e)

    def test_connection(self, bucket: str):
        try:
            self.s3.list_objects_v2(Bucket=bucket, MaxKeys=1)
            return True
        except Exception as e:
            print(f"AWS connection test failed: {e}")
            return False
