# Inference Container

This folder includes the inference container.  build-push-create.ipynb notebook walks you thru the process of building Docker image, pushing the image to Amazon ECR, and creating an Amazon SageMaker endpoint. 

Container folder has Docker and shell script files to configure Docker image with all necessary software packages. In addition, it has ProtCNN folder which includes Python scripts to run a web server (predictor.py), define ProtCNN model (ProtCNN.py), and process incoming requests (utility_methods.py). 

