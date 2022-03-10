import sagemaker


def main(
    checkpoint_location="s3://ai-cv/hades/ucn-weights/eitl-checkpoints",
):
    instance_type = "ml.g4dn.2xlarge"
    train_data_path = "s3://ai-cv/hades/toei-all-train"

    output_path = "s3://ai-cv/hades/output"
    role = "arn:aws:iam::842977422697:role/SageMaker_Cinnamon_role"

    # Create Sagemaker session
    sess = sagemaker.Session()
    account = sess.boto_session.client("sts").get_caller_identity()["Account"]
    region = sess.boto_session.region_name

    repo_name = "sagemaker-hades"
    image_tag = "hades01"
    image_name = "{}.dkr.ecr.{}.amazonaws.com/{}:{}".format(account, region, repo_name, image_tag)
    base_job_name = "hades-sagemaker-toei"

    train_data_channel = sagemaker.inputs.TrainingInput(
        train_data_path,
        distribution="FullyReplicated",
        s3_data_type="S3Prefix")

    # Handle end-to-end Amazon SageMaker training and deployment tasks
    estimator = sagemaker.estimator.Estimator(
        image_uri=image_name,
        base_job_name=base_job_name,
        role=role,
        input_mode="File",
        instance_count=1,
        instance_type=instance_type,
        output_path=output_path,
        checkpoint_s3_uri=checkpoint_location,  # checkpoint_s3_uri : where on S3 will sync the checkpoint of the model (which on checkpoint_local_path). 
                                                # If the training job is resumed, sagemaker will go here to get the checkpoint 
                                                # and upload it to checkpoint_local_path of the new session. 

        max_run=4*86400,
        volume_size=125,
        use_spot_instances=True,
        max_wait=5*86400,
        sagemaker_session=sess)

    estimator.fit({"train": train_data_channel}, wait=False)


if __name__ == "__main__":
    main()
