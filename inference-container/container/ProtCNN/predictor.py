import tensorflow as tf
import os
import json
import boto3
from io import StringIO
import flask
import pandas as pd
import numpy as np

# Import model and utility methods
from ProtCNN import ProtCNN
import utility_methods as um

tf.config.experimental_run_functions_eagerly(True)

# The total number of classes to be predicted
unique_classes = 17929
# The maximun length a sequence can have. Shorter sequences are padded with zeros
max_len = 2037
# 25 amino_acids (indexed from 1) plus 1 to account for zero padding 
amino_acids = 26

# The flask app for serving predictions
app = flask.Flask(__name__)

print("Building model...")
prot_cnn = ProtCNN(unique_classes)
prot_cnn.build((None, max_len, amino_acids))
print("Done building model...")

# Restoring model weights
prefix = '/opt/ml'
checkpoint_dir = os.path.join(prefix, 'model')
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

print("Restoring model...")
checkpoint = tf.train.Checkpoint(prot_cnn=prot_cnn)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=1)
status = checkpoint.restore(manager.latest_checkpoint)
print("Done restoring model...")

"""
unique_families.csv file includes unique family_accession and family_id attributes. They are needed to convert 
    model predictions to the proper family accessions and IDs.
"""
unique_families = pd.read_csv("./unique_families.csv")
family_accessions = list(unique_families["family_accession"])
family_ids = list(unique_families["family_id"])
    
@app.route('/ping', methods=['GET'])
def ping():
    status = 200
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    
    # A placeholder for input data
    data = None
    # A list to store input data
    data_list = []
    
    if flask.request.content_type == 'text/csv':
        # Decode request
        data = flask.request.data.decode('utf-8')
        
        """
        We have two types of requests. The first type is originated by Batch Transform job, and the
            other is from AWS Lambda. Batch Transform does not include "\r\n" in requests while AWS Lambda does. 
            So, we use this fact to distinguish the source of requests. 
        """
        
        # Requests from Batch Transform job
        if len(data.split("\r\n")[:-1]) == 0:
            temp = um.sequence_to_ID(data.strip())
            data_list.append(temp)
        
        # Requests from AWS Lambda
        else:
            
            # Loop over input sequences and convert them to IDs
            for element in data.split("\r\n")[:-1]:
                temp = um.sequence_to_ID(element)
                data_list.append(temp)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')
    
    print('Invoked with {} records'.format(len(data_list)))
    
    # Convert list to array, pad sequences, and one-hot encode them
    data_array = um.seq_to_np(data_list)
    data_array = um.one_hot(data_array)
    
    print("Making predictions...")
    predictions = prot_cnn(data_array, training=False)
    predictions = predictions.numpy()
    
    indices, confidences = np.argmax(predictions, axis=1), np.amax(predictions, axis=1)
    
    # Store each prediction in a dictionary
    dic = {}
    for index, confidence, i in zip(indices, confidences, range(predictions.shape[0])):
        predicted_fam_acc = family_accessions[index]
        predicted_fam_id = family_ids[index]
        dic[i] = (predicted_fam_acc, predicted_fam_id, confidence)
    
    out = StringIO()
    pd.DataFrame({'results':dic}).to_csv(out, header=False, index=False)
    result = out.getvalue()
    return flask.Response(response=result, status=200, mimetype='text/csv')
