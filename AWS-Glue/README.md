# AWS-Glue

This folder includes Pfam data processing code. The code is presented as an IPython notebook. The notebook assumes you have an access to an AWS Glue database. The database should have three tables which include metadata for train, validation, and test data. 

The notebook also assumes that the freq_df.csv and dict_class.csv files are stored in Amazon S3. You can upload the files to the notebook instance and access them directly. freq_df.csv includes frequencies of sequence letters, and dict_class.csv has all unique family accessions. The files are used to convert letters and accessions to their corresponding IDs. For more information, please refer to the blog.

TO DO:
Add the link to the blog
