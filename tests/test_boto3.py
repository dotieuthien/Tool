from sys import prefix
import boto3


def test_client_resource():
    # CLIENT
    # you directly interact with response dictionary from a deserialized API response
    s3_client = boto3.client(
        's3',
        region_name='us-west-2',
        aws_access_key_id='AKIA4IRK6VVURMRAHE4W',
        aws_secret_access_key='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'
        )

    for s3_object in s3_client.list_objects(Bucket='ai-cv', Prefix='labeling')['Contents']:
        print(s3_object['Key'])

    # RESOURCE
    # you interact with standard Python classes and 
    # objects rather than raw response dictionaries
    s3_resource = boto3.resource(
        's3',
        region_name='us-west-2',
        aws_access_key_id='AKIA4IRK6VVURMRAHE4W',
        aws_secret_access_key='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'
        )
    bucket = s3_resource.Bucket(name='ai-cv')
    resource_keys = list(s3_object.key for s3_object in bucket.objects.filter(Prefix='labeling'))
    print(resource_keys)

if __name__ == "__main__":
    test_client_resource()

