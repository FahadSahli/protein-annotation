//import Amplify from '@aws-amplify/core';
import Storage from '@aws-amplify/storage';
import Amplify, { Auth } from 'aws-amplify';

export function configureAmplify() {
  Amplify.configure(
  {
   Auth: {
     identityPoolId: "us-east-1:8f4d2854-1619-41bd-a8b5-df90a47c3646",
     region: "us-east-1",
     userPoolId: "us-east-1_A8aY37N7r",
     userPoolWebClientId: "3r3k1hvqjan44juj4gos9osvsp",
     mandatorySignIn: false,
    },
  Storage: { 
     bucket: "intern-project-web-tier-uploads",
     region: "us-east-1",
     identityPoolId: "us-east-1:8f4d2854-1619-41bd-a8b5-df90a47c3646"
    }
  }
 );
}
//Configure Storage with S3 bucket information
export function SetS3Config(bucket, level){
   Storage.configure({ 
          bucket: bucket,
          level: level,
          region: 'us-east-1',  
          identityPoolId: process.env.REACT_APP_identityPoolId 
       });
}