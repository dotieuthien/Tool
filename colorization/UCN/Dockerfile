# Get base image
FROM base

ARG py_version=3
RUN test $py_version || exit 1

# Copy the code
RUN rm -rf /opt/ml/code/
COPY . /opt/ml/code/
ENV PATH="/opt/ml/code:${PATH}"

# Specify Working dir
WORKDIR /opt/ml/code/

# Start framework
ENTRYPOINT ["python", "sagemaker_entry.py", "--config", "sagemaker_config.ini"]
