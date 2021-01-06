# YAML Templates

This folder includes two YAML templates. training-template.yaml has resources for data processing and model training, tuning, and deployment. Data processing is illustrated in protein-annotation/AWS-Glue/, model training is illustrated in protein-annotation/training/, and model deployment is illustrated in protein-annotation/inference-container/. <br>
service-integration-template.yaml template integrates deployed model with other AWS services such as Amazon S3, Amazon Cognito, and AWS AppSync. After you deploy the stack, you need to do make some configurations with Amazon S3. First, you should add a trigger for AWS Lambda: <br>
1.	Go to the created Amazon S3 bucket’s “Properties”
2.	Then, go to “Event notifications”, click on “Create event notification”
3.	Choose a name for the event
4.	Check the box next to “All object create events” under “Event types”
5.	Under “Destination”, choose “Lambda function”, “Choose from your Lambda functions”, then pick the created function
6.	Finally, click on “Save changes” button <br>
Second, go to “Permissions”, “Cross-origin resource sharing (CORS)”, click on “Edit” button, and past the following: <br>
[ <br>
    { <br>
        "AllowedHeaders": [ <br>
            "*" <br>
        ], <br>
        "AllowedMethods": [ <br>
            "PUT", <br>
            "POST", <br>
            "DELETE" <br>
        ], <br>
        "AllowedOrigins": [ <br>
            "*" <br>
        ], <br>
        "ExposeHeaders": [] <br>
    } <br>
] <br>
<br>
Also, the template does not host the web application on AWS Amplify. As a result, you need to configure your AWS Cloud9 environment. The details can be found in protein-annotation/front-end/ReadMe.txt.


