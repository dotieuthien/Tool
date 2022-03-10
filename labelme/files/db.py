import pymongo
import boto3
import botocore


class DataManager:
    def __init__(self, 
    db_url, 
    aws_access_key_id, 
    aws_secret_access_key, 
    s3_region_name, 
    s3_bucket):
        """[summary]

        Args:
            db_url ([type]): [description]
            aws_access_key_id ([type]): [description]
            aws_secret_access_key ([type]): [description]
            s3_region_name ([type]): [description]
            s3_bucket ([type]): [description]
        """
        self.s3_bucket = s3_bucket
        
        # Check db connection
        try:
            client = pymongo.MongoClient(
                db_url,
                serverSelectionTimeoutMS=10 # 10ms
                )
            client.server_info()
            print('Successfully connect to database')
        except pymongo.errors.ServerSelectionTimeoutError as err:
            print(err)

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
                self.s3 = s3
        except:
            print('Cannot connect to s3')

    def pull_db():
        pass

    def push_db():
        pass

    def pull_data_from_s3(self):
        self.s3.Bucket('ai-cv').download_file('labeling/0001.tga', '0001.tga')

    def push_data_to_s3(self):
        self.s3.Bucket('ai-cv').upload_file(r'D:\Tool\data\0001.tga', 'labeling/0001.tga')

    def delete_data_from_s3(self):
        self.s3.Object('ai-cv', "sketch_0005__sketch_0008.json").delete()


if __name__ == "__main__":
    # # Client for a MongoDB instance
    # myclient = pymongo.MongoClient('localhost', 27017)
    # print(myclient.list_database_names())
    # mydb = myclient["colorization-labeling-tool"]
    # toei_collection = mydb["toei"]
    
    # mylist = [
    # { "name": "Amy", "address": "Apple st 652"},
    # { "name": "Hannah", "address": "Mountain 21"},
    # { "name": "Michael", "address": "Valley 345"},
    # { "name": "Sandy", "address": "Ocean blvd 2"},
    # { "name": "Betty", "address": "Green Grass 1"},
    # { "name": "Richard", "address": "Sky st 331"},
    # { "name": "Susan", "address": "One way 98"},
    # { "name": "Vicky", "address": "Yellow Garden 2"},
    # { "name": "Ben", "address": "Park Lane 38"},
    # { "name": "William", "address": "Central st 954"},
    # { "name": "Chuck", "address": "Main Road 989"},
    # { "name": "Viola", "address": "Sideway 1633"}
    # ]

    # x = toei_collection.insert_many(mylist)
    # print(x.inserted_ids)

    # geek_collection = mydb["geek"]

    # mylist = [
    # { "name": "Amy", "address": "Apple st 652"},
    # { "name": "Hannah", "address": "Mountain 21"},
    # { "name": "Michael", "address": "Valley 345"},
    # { "name": "Sandy", "address": "Ocean blvd 2"},
    # { "name": "Betty", "address": "Green Grass 1"},
    # { "name": "Richard", "address": "Sky st 331"},
    # { "name": "Susan", "address": "One way 98"},
    # { "name": "Vicky", "address": "Yellow Garden 2"},
    # { "name": "Ben", "address": "Park Lane 38"},
    # { "name": "William", "address": "Central st 954"},
    # { "name": "Chuck", "address": "Main Road 989"},
    # { "name": "Viola", "address": "Sideway 1633"}
    # ]

    # y = geek_collection.insert_many(mylist)
    # print(y.inserted_ids)

    # s3 = boto3.resource(
    # service_name='s3',
    # region_name='us-west-2',
    # aws_access_key_id='AKIA4IRK6VVURMRAHE4W',
    # aws_secret_access_key='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'
    # )
    # # Print out bucket names
    # for bucket in s3.buckets.all():
    #     print(bucket.name)

    # for id, obj in enumerate(s3.Bucket('ai-cv').objects.all()):
    #     print(obj)
    #     if id == 100:
    #         break

    dm = DataManager(
        'mongodb://localhost:27017/', 
        'AKIA4IRK6VVURMRAHE4W', 
        'wKEtRq6YgjQQNpqmW8+3t9GeE8z/9DGKiSBorrPo',
        'us-west-2',
        'ai-cv')
    dm.delete_data_from_s3()