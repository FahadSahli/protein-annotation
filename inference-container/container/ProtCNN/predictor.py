import tensorflow as tf
import os
import json
import boto3
from io import StringIO
import flask
import pandas as pd
from ProtCNN import ProtCNN
import utility_methods as um

tf.config.experimental_run_functions_eagerly(True)
number_of_unique_classes = 17929
max_len = 2037
# 25 amino_acids (indexed from 1) plus 1 to account for zero padding 
amino_acids = 26

# The flask app for serving predictions
app = flask.Flask(__name__)

print("Building model...")
prot_cnn = ProtCNN(number_of_unique_classes)
prot_cnn.build((None, max_len, amino_acids))
print("Done building model...")

prefix = '/opt/ml'
checkpoint_dir = os.path.join(prefix, 'model')
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

print("Restoring model...")
checkpoint = tf.train.Checkpoint(prot_cnn=prot_cnn)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=1)
status = checkpoint.restore(manager.latest_checkpoint)
print("Done restoring model...")

@app.route('/ping', methods=['GET'])
def ping():
    status = 200
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    data = None
    if flask.request.content_type == 'text/csv':
        # Decode request
        data = flask.request.data.decode('utf-8')
        
        # Loop over input sequences and convert them to IDs
        data_list = []
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
    
    # Store each prediction in a dictionary
    dic = {}
    for i in range(predictions.shape[0]):
        dic[i] = list(predictions[i])
    
    out = StringIO()
    pd.DataFrame({'results':dic}).to_csv(out, header=False, index=False)
    result = out.getvalue()
    return flask.Response(response=result, status=200, mimetype='text/csv')