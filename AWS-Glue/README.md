# AWS-Glue

This folder includes Pfam data processing code. The code is presented as an IPython notebook. The notebook assumes you have an access to an AWS Glue database. The database should have three tables which include metadata for train, validation, and test data. 

The notebook also assumes that freq_df.csv and dict_class.csv files are stored in Amazon S3. freq_df.csv includes frequencies of sequence letters, and dict_class.csv has all unique family accessions. The files are used to convert letters and accessions to their corresponding IDs. For more information, please refer to the blog.

To use the notebook, you need to have crawlers connected to the data, and you also need to create a notebook instance from AWS Glue console. To create a crawler for each data file (e.g., train, validation, or test), do the following:
1. Create a crawler
2. Choose a name for the crawler, then click "next"
3. For "Crawler source type", choose "Data stores"
4. For "Repeat crawls of S3 data stores", choose "Crawl all folders", then click "next"
5. For "Choose a data store", choose "S3"
6. For "Crawl data in", choose "Specified path in my account"
7. For "Include path", choose the path the includes a data file, then click "next"
8. For "Add another data store", choose "No", then click "next"
9. For IAM role, choose the role which is the one of the development endpoint, then click "next"
10. For "Frequency", choose "Run on demand", then click "next"
11. For "Database", click on "Add database" and specify its name if you have not created one yet, otherwise, choose the one you created, then click "next"
12. Review your configurations, then click "Finish"
13. When the crawler is on ready state, run it

Regarding the notebook instance, the creation process is as the following:
1. Choose a name for the notebook
2. Choose the development endpoint created by the AWS CloudFormation stack
3. Choose the IAM role which is the one of the development endpoint
4. Choose the VPC created by the AWS CloudFormation stack
5. Choose the public subnet of the VPC
6. Choose the security group which is the one of the development endpoint
7. Review your configurations, then click "Create notebook"


TO DO:
Add the link to the blog
Add link to AWS CloudFormation

