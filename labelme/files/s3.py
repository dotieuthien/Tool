import boto3
import os


# TODO: boto3 client have an operation is .generate_presigned_url() enables to 
# give QA the credential for a set period of time

# TODO: ACL


def init_s3_object(
        s3_bucket,
        s3_region_name,
        aws_access_key_id,
        aws_secret_access_key):
    """Create S3 object

    Args:
        s3_bucket (_type_): _description_
        s3_region_name (_type_): _description_
        aws_access_key_id (_type_): _description_
        aws_secret_access_key (_type_): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    # Check s3 bucket connection
    try:
        s3 = boto3.resource(
                service_name='s3',
                region_name=s3_region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
                )

        bucket_names = []

        for bucket in s3.buckets.all():
            bucket_names.append(bucket.name)

        if s3_bucket not in bucket_names:
            raise Exception('Cannot file the bucket')
        else:
            print('Successfully connect to s3')
            # Download latest checkpoint
            download_latest_checkpoint(s3)

            return s3

    except:
        print('Cannot connect to s3')
        return


def download_latest_checkpoint(s3_resource, save_dir='colorization/UCN/pretrained-weights'):
    checkpoint_paths = []

    for id, obj in enumerate(s3_resource.Bucket('ai-cv').objects.filter(Prefix="hades/ucn-weights/eitl-checkpoints")):
        checkpoint_paths.append(obj.key)
        latest_path = obj.key
        
    # Download and save checkpoint
    save_path = os.path.join(save_dir, os.path.basename(latest_path))
    print('Latest checkpoint downloading...')
    # download_file_from_s3(s3_resource, 'ai-cv', latest_path, save_path)
    print('Download complete')


def download_file_from_s3(s3_resource, bucket_name, file_uri, save_path):
    """Download image from s3 to label

    Args:
        bucket_name (str):
        file_uri (str): relative s3 path to the folder
        save_path (str): path to store file
    """
    try:
        s3_resource.Bucket(bucket_name).download_file(file_uri, save_path)
    except:
        print(f"Error {file_uri} cannot be retrieved")


def upload_file_to_s3(s3_resource, bucket_name, local_path_file, file_uri):
    """Upload label to s3

    Args:
        bucket_name (str):
        local_path_file (str): path to file in local machine
        file_uri (str): path to store file in s3
    """
    try:
        # Upload via Bucket object
        s3_resource.Bucket(bucket_name).upload_file(local_path_file, file_uri)
    except:
        print(f"Error {file_uri} cannot be uploaded")