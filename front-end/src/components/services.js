//import Amplify from '@aws-amplify/core';
import Storage from '@aws-amplify/storage';

//Configure Storage with S3 bucket information
export function SetS3Config(bucket, level, region){
   Storage.configure({ 
          bucket: bucket,
          level: level,
          region: region,  
          identityPoolId: process.env.REACT_APP_identityPoolId 
       });
}
