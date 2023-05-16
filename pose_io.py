import urllib.request
import boto3

class downloader:
    def __init__(self, url_link, file_addr) -> None:
        self.url_link = url_link
        self.file_addr = file_addr

    def download(self):
        urllib.request.urlretrieve(self.url_link, self.file_addr) 

class uploader:
    def __init__(self, file_addr) -> None:
        self.file_addr = file_addr
        self.out_file_name = file_addr.split('/')[2]
        self.AWS_ACCESS_KEY_ID = 'nKureGoQBu0QD4LV'
        self.AWS_SECRET_ACCESS_KEY = 'LNd0MMrfoFSstuHuCiAuGPkLLIBx745W'

    def upload(self):
        session = boto3.Session(region_name='default')
        bucket_name = 'uploads'

        client = session.client('s3', 
                        endpoint_url='https://hedieh-p1.darkube.app',
                        aws_access_key_id=self.AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY)
        
        client.upload_file(self.file_addr, bucket_name, self.out_file_name)

