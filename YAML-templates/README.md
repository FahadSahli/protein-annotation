# YAML Templates

This folder includes two YAML templates. [`training-template.yaml`](./training-template.yaml) has resources for data processing and model training, tuning, and deployment. Data processing is illustrated in [`protein-annotation/AWS-Glue/`](https://github.com/FahadSahli/protein-annotation/tree/main/AWS-Glue), model training is illustrated in [`protein-annotation/training/`](https://github.com/FahadSahli/protein-annotation/tree/main/training), and model deployment is illustrated in [`protein-annotation/inference-container/`](https://github.com/FahadSahli/protein-annotation/tree/main/inference-container). <br> 
<br>
[`service-integration-template.yaml`](./service-integration-template.yaml) template integrates deployed model with other AWS services such as Amazon S3, Amazon Cognito, and AWS AppSync. The template has some restricts. All resource names are dependent on the template’s name. Some resources do not accept hyphens (-) in their names. So, the template’s name must not contain hyphens. There are two required parameters which are URI of pushed Inference Container and URI of trained model’s artifacts. After you deploy the template, you need to make some configurations on Amazon S3. First, you should add a trigger for AWS Lambda: <br>
1.	Go to the created Amazon S3 bucket’s “Properties”
2.	Then, go to “Event notifications”, click on “Create event notification”
3.	Choose a name for the event
4.	Check the box next to "Put" under “All object create events” under “Event types”
5.	Under “Destination”, choose “Lambda function”, “Choose from your Lambda functions”, then pick the created function
6.	Finally, click on “Save changes” button <br>
<br>
Second, go to “Permissions”, “Cross-origin resource sharing (CORS)”, click on “Edit” button, and then past the following: <br>

```
[
    { 
        "AllowedHeaders": [
            "*"
        ], 
        "AllowedMethods": [
            "PUT",
            "POST",
            "DELETE"
        ], 
        "AllowedOrigins": [
            "*"
        ],
        "ExposeHeaders": [] 
    }
]
```

Also, You need to edit some parameters in the front-end code based on the output from the template. The template does not host the web application on AWS Amplify. As a result, you need to configure your AWS Cloud9 environment. The details can be found in [`protein-annotation/front-end/ReadMe.txt`](https://github.com/FahadSahli/protein-annotation/blob/main/front-end/README.md).



