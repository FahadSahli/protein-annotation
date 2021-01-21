# Front End
This folder includes all source code needed to run the front end ReactJS application. However, there are some parameters you need to populate based on the stack’s output. Go to the file [`./src/aws-exports.js`](./src/aws-exports.js) and add the parameters from the stack’s output. Then, go to line 28 of the file [`./src/components/Home.js`](./src/components/Home.js) and add the name of created bucket and the region where you deploy the resources (e.g., ``` SetS3Config("bucket ", "private", "regoin"); ```). <br> 
<br> 

As the YAML template does not host the application on AWS Amplify, you need to configure your AWS Cloud9 environment. What you need is to run the following commands:

```
npm install aws-amplify 
npm install aws-amplify @aws-amplify/ui-react 
npm install aws-amplify @aws-amplify/storage 
npm install aws-amplify @aws-amplify/core 
npm install aws-amplify @material-ui/core 
npm install aws-amplify-react --save
npm install react-router-dom 
cd front-end 
npm install 
npm start 
```

Now, you have a working demo of Protein Annotation.
