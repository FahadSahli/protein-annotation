# Front End
This folder includes all source code needed to run the front end ReactJS application. However, there are some parameters you need to populate based on the stack’s output. Go to the file protein-annotation/front-end/src/aws-exports.js and add the parameters from the stack’s output. Then, go to line 28 of the file protein-annotation/front-end/src/components/Home.js and add the name of created bucket and the region where you deploy the resources (e.g., SetS3Config("bucket ", "private", "regoin");). <br> 
<br> 

As the YAML template does not host the application on AWS Amplify, you need to configure your AWS Cloud9 environment. What you need is to run the following commands:
<br> 
<br> 
npm install aws-amplify <br> 
npm install aws-amplify @aws-amplify/ui-react <br> 
npm install aws-amplify @aws-amplify/storage <br> 
npm install aws-amplify @aws-amplify/core <br> 
npm install aws-amplify @material-ui/core <br> 
npm install aws-amplify-react --save <br> 
npm install react-router-dom <br> 
cd front-end <br> 
npm install <br> 
npm start <br> 
