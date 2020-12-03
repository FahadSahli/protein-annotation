import tensorflow as tf
import numpy as np
import codecs
import csv
import boto3
import pandas as pd

# S3 client to read files stored in S3
client = boto3.client("s3")

"""
freq_df.csv contains frequencies of letters accross the whole data. The letters are sorted in decreasing order 
    based on their frequencies. The content of the file is stored in the df_freq datafram. The index of a letter 
    is used as an ID of that letter after adding 1 to it. For example, the letter at index 2 is givien the ID 3 (i.e., 2+1).
    
The datafram has two columns which are feature and frequency. The feature column consists of the letters.
"""
df_freq = pd.read_csv('./freq_df.csv')

def sequence_to_ID(sequence):
    """
    This method converts letters in a sequence to their IDs.
    Inputes:
        1. A sequence of letters.
        
    Outputs:
        1. A list of IDs.
    """
    return list(map(char_to_ID, list(sequence)))

def char_to_ID(char):
    return df_freq.index[df_freq['feature'] == char.lower()].tolist()[0] + 1


def seq_to_np(inp_list, max_len=2037):
    """
    This method converts input sequences into numpy arrays.
    Inputes:
        1. A list of sequences.
        
    Outputs:
        1. A an array of padded sequences.
    """
    
    # Get the first element and pad it
    array = np.asarray(inp_list[0], dtype=np.intc)
    array = tf.keras.preprocessing.sequence.pad_sequences([array], maxlen=max_len, dtype='int32', padding='post', value=0)
    
    # Loop over the rest of the input list
    for i in range(1, len(inp_list)):
        temp = np.asarray(inp_list[i], dtype=np.intc)
        temp = tf.keras.preprocessing.sequence.pad_sequences([temp], maxlen=max_len, dtype='int32', padding='post', value=0)
        array = np.concatenate((array, temp), axis=0) 
    
    return array

def one_hot(final_sequence, nb_classes = 26):
    return tf.keras.utils.to_categorical(final_sequence, nb_classes)

"""
def split_sequence(sequence):
    
    separated_seq = ''
    for i in range(len(sequence)):
        separated_seq = separated_seq + ' ' + sequence[i]
    return separated_seq
    
def read_csv_from_s3(bucket_name, key, column):
    data = client.get_object(Bucket=bucket_name, Key=key)
    read_data = []
    
    for row in csv.DictReader(codecs.getreader("utf-8")(data["Body"])):
        read_data.append(row[column])
        
    return np.array(read_data)

def input_y_to_np(inp_df):
    
    array = np.asarray(inp_df[0], dtype=np.intc)
    temp = np.asarray(inp_df[1], dtype=np.intc)
    
    array = np.concatenate(([array], [temp]), axis=0)
    
    for i in range(2, len(inp_df)):
        temp = np.asarray(inp_df[i], dtype=np.intc)
        array = np.concatenate((array, [temp]), axis=0)
    
    return array
    
def get_categorical(y_train, number_of_unique_classes):
    
    Y_train = []
    for label in y_train:
        label_ = tf.keras.utils.to_categorical(label , number_of_unique_classes) 
        Y_train.append(label_)
    Y_train = np.array(Y_train)
    return Y_train
"""