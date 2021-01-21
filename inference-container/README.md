# Inference Container

This folder includes the inference container.  [`build-push-create.ipynb`](./build-push-create.ipynb) notebook walks you thru the process of building Docker image, pushing the image to Amazon ECR, and creating an Amazon SageMaker endpoint. The file [`sample-test.csv`](./sample-test.csv) has five test samples to be used when the created endpoint is invoked within the [`build-push-create.ipynb`](./build-push-create.ipynb) notebook.

Container folder has Docker and shell script files to configure Docker image with all necessary software packages. In addition, it has ProtCNN folder which includes Python scripts to run a web server ([`predictor.py`](./container/ProtCNN/predictor.py)), define ProtCNN model ([`ProtCNN.py`](./container/ProtCNN/ProtCNN.py)), and process incoming requests ([`utility_methods.py`](./container/ProtCNN/utility_methods.py)). 

Most of the files in this folder are based on [`this repository`](https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/scikit_bring_your_own).

