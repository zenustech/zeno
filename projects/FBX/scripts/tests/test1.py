import os
import sys
import oss2
import json


class ConnectOss(object):
    def __init__(self, access_id, access_key, bucket_name):
        self.auth = oss2.Auth(access_id, access_key)
        self.endpoint = 'https://oss-cn-shenzhen.aliyuncs.com'
        self.bucket = oss2.Bucket(self.auth, self.endpoint, bucket_name=bucket_name)

    def get_bucket_list(self):
        """list all bucket_name under current endpoint"""
        service = oss2.Service(self.auth, self.endpoint)
        bucket_list = [b.name for b in oss2.BucketIterator(service)]
        return bucket_list

    def get_all_file(self, prefix):
        """get all file by specific prefix"""
        for i in oss2.ObjectIterator(self.bucket, prefix=prefix):
            print("file", i.key)

    def read_file(self, path):
        try:
            file_info = self.bucket.get_object(path).read()
            return file_info
        except Exception as e:
            print(e, 'File does not exists')

    def download_file(self, path, save_path):
        result = self.bucket.get_object_to_file(path, save_path)
        if result.status == 200:
            print('Download Complete')

    def upload_file(self, path, local_path):
        result = self.bucket.put_object_from_file(path, local_path)
        if result.status == 200:
            print('Upload Success')


def main():
    f = open("c:/src/config.json")

    config_data = json.load(f)
    access_id = config_data["access_id"]
    access_key = config_data["access_key"]
    bucket_name = config_data["bucket_name"]

    print("OSS Config access_id", access_id)
    print("OSS Config access_key", access_key)
    print("OSS Config bucket_name", bucket_name)

    oss = ConnectOss(access_id, access_key, bucket_name)

    # oss.upload_file("test/upload/log.rar", "C:/src/zeno/build/log.rar")
    # oss.download_file("test/upload/log.rar", "C:/src/log.rar")
    # print("file", oss.read_file("test/upload/log.rar"))
    oss.get_all_file("test/upload")


if __name__ == '__main__':
    main()
