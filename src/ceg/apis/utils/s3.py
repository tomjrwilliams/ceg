import os
import boto3 # type: ignore

def iter_objects(bucket: str):
    s3 = boto3.resource('s3', **get_credentials)
    b = s3.Bucket(bucket)
    for obj in b.objects.all():
        yield obj
        # obj.key, etc.

def get_credentials():
    return dict(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key= os.environ["AWS_SECRET_ACCESS_KEY"]
    )

def put_object(
    bucket: str, 
    key: str, 
    obj: bytes,
    encryption_key: str | None = None,
    encryption_algo: str | None = None,
):
    s3 = boto3.client('s3', **get_credentials())
    return s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=obj,
    )

def get_object(
    bucket: str,
    key: str,
):
    s3 = boto3.client('s3', **get_credentials())
    response = s3.get_object(
        Bucket=bucket,
        Key=key,
    )
    try:
        return response["Body"].read()
    except:
        raise ValueError(response)


def filter_objects_by_last_modified(bucket: str):
    s3 = boto3.client("s3")
    s3_paginator = s3.get_paginator('list_objects_v2')
    s3_iterator = s3_paginator.paginate(Bucket=bucket)
    filtered_iterator = s3_iterator.search(
        "Contents[?to_string(LastModified)>='\"2022-01-05 08:05:37+00:00\"'].Key"
    )
    for key_data in filtered_iterator:
        yield key_data