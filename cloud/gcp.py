from google.cloud import storage
import json

class GCPConnector:
    def __init__(self, credentials_json: str):
        # credentials_json is the raw content of the service account file
        creds_dict = json.loads(credentials_json)
        self.client = storage.Client.from_service_account_info(creds_dict)

    def upload_file(self, local_path: str, bucket_name: str, blob_name: str):
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            return True, "Upload successful"
        except Exception as e:
            return False, str(e)
