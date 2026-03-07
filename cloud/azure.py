from azure.storage.blob import BlobServiceClient

class AzureConnector:
    def __init__(self, credentials: dict):
        if credentials.get('connection_string'):
            self.client = BlobServiceClient.from_connection_string(credentials['connection_string'])
        else:
            self.client = BlobServiceClient(
                account_url=f"https://{credentials['account_name']}.blob.core.windows.net",
                credential=credentials['key']
            )

    def upload_file(self, local_path: str, container: str, blob_name: str):
        try:
            blob_client = self.client.get_blob_client(container=container, blob=blob_name)
            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            return True, "Upload successful"
        except Exception as e:
            return False, str(e)
