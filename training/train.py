import argparse
from tensorflow.keras import layers
import tensorflow as tf
import logging
import pandas as pd
import numpy as np
import os
import random
import time
import json
import codecs
import csv
import boto3
from io import StringIO

# To ensure TF runs eagerly
tf.config.experimental_run_functions_eagerly(True)

logging.getLogger().setLevel(logging.INFO)

# S3 client to read files stored in S3
client = boto3.client("s3")

# A class that defines the model
class ProtCNN(tf.keras.Model):
    def __init__(self, unique_classes):
        
        super(ProtCNN, self).__init__()
        self.unique_classes = unique_classes
        
        # Define the layers of the model
        
        self.conv1d_1 = layers.Conv1D(32, 1, strides=1, padding='valid', name='conv1d_1', 
                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))
        self.max_pool_1 = layers.MaxPool1D(pool_size=2)
        self.batch_norm_1 = layers.BatchNormalization(axis=2, name='batch_norm_1')
        
        self.activation_1 = layers.Activation('relu', name='activation_1')
        self.batch_norm_2 = layers.BatchNormalization(axis=2, name='batch_norm_2')
        self.activation_2 = layers.Activation('relu', name='activation_2')
        
        
        self.conv1d_2 = layers.Conv1D(32, 1, strides=1, padding='valid', name='conv1d_2', 
                           kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))
        self.batch_norm_3 = layers.BatchNormalization(axis=2, name='batch_norm_3')
        self.activation_3 = layers.Activation('relu', name='activation_3')
        
        
        self.conv1d_3 = layers.Conv1D(32, 1, strides=1, padding='valid', name='conv1d_3',
                                 kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))
        self.dropout_1 = layers.Dropout(0.5, name='dropout_1')
        self.max_pool_2 = layers.MaxPool1D(pool_size=2)
        
        
        self.conv1d_4 = layers.Conv1D(32,  1, strides=1, padding='valid', name='conv1d_4',
                                 kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))
        self.dropout_2 = layers.Dropout(0.5, name='dropout_2')
        self.max_pool_3 = layers.MaxPool1D(pool_size=2)
        
        
        self.added = layers.Add()
        
        self.activation_4 = layers.Activation('relu',name='activation_4')
        self.dropout_3 = layers.Dropout(0.2, name='dropout_3')
        
        self.batch_norm_4 = layers.BatchNormalization(axis=2, name='batch_norm_4')
        self.activation_5 = layers.Activation('relu', name='activation_5')
        self.dropout_4 = layers.Dropout(0.5, name='dropout_4')
        
        self.flatten = layers.Flatten(name='flatten')
        self.dense = layers.Dense(self.unique_classes, name="dense",
                          kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))
        self.final_output = layers.Activation('softmax', name='output')
        
        
    def call(self, x):
        """
        This method is called whenever the model is invoked, either for training or testing.
        x is the input, and it has a shape of (batch, max_len, amino_acids).
        """
        conv1d_1 = self.conv1d_1(x)
        max_pool_1 = self.max_pool_1(conv1d_1)
        batch_norm_1 = self.batch_norm_1(max_pool_1)

        activation_1 = self.activation_1(batch_norm_1)
        batch_norm_2 = self.batch_norm_2(activation_1)
        activation_2 = self.activation_2(batch_norm_2)

        conv1d_2 = self.conv1d_2(activation_2)
        batch_norm_3 = self.batch_norm_3(conv1d_2)
        activation_3 = self.activation_3(batch_norm_3)

        conv1d_3 = self.conv1d_3(activation_3)
        dropout_1 = self.dropout_1(conv1d_3)
        max_pool_2 = self.max_pool_2(dropout_1)

        conv1d_4 = self.conv1d_4(activation_1)
        dropout_2 = self.dropout_2(conv1d_4)
        max_pool_3 = self.max_pool_3(dropout_2)

        added = self.added([max_pool_2, max_pool_3])

        activation_4 = self.activation_4(added)
        dropout_3 = self.dropout_3(activation_4)

        batch_norm_4 = self.batch_norm_4(dropout_3)
        activation_5 = self.activation_5(batch_norm_4)
        dropout_4 = self.dropout_4(activation_5)

        flatten = self.flatten(dropout_4)
        dense = self.dense(flatten)
        output = self.final_output(dense)
        
        return output
    
def inp_df_to_np(inp_df):
    """
    This method converts input sequences into numpy arrays.
    
    Each entry in inp_df dataframe is a string of the form "[7, 10, 5, 1, 19, 14, 10, 10, 3, ...]",
        where the numbers act as IDs for the 25 letters that makeup a sequence. The brackets are part
        of the string and should be removed.
        
    The method returns an array with shape (batch, max_len)
    """
    
    # Get the first entry in inp_df, remove brackets, and split letters
    splited = inp_df.iloc[0].strip("[]").split(", ")
    
    # Convert the list of IDs to numpy array of integers
    array = np.asarray([int(x) for x in splited], dtype=np.intc)
    
    # Pad the array with zeros
    array = tf.keras.preprocessing.sequence.pad_sequences([array], maxlen=max_len, dtype='int32', padding='post', value=0)
    
    # Loop over the rest of the entries in inp_df
    for i in range(1, len(inp_df)):
        splited = inp_df.iloc[i].strip("[]").split(", ")
        temp = np.asarray([int(x) for x in splited], dtype=np.intc)
        temp = tf.keras.preprocessing.sequence.pad_sequences([temp], maxlen=max_len, dtype='int32', padding='post', value=0)
        array = np.concatenate((array, temp), axis=0) 
    
    return array

def loss_function(real, pred):
    loss_ = loss_object(real, pred)

    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ):
    """
    This method passes data to the model, computes loss, and updates model parameters.
    Inputes:
        1. inp, a batch of training sequences.
        2. targ, a batch of target labels.
        
    Outputs:
        1. batch_loss.
        2. m.result().numpy() which is the batch accuracy.
    """
    global prot_cnn, optimizer, loss_object
    
    loss = 0
    
    # A Keras object to calculate accuracy
    m = tf.keras.metrics.CategoricalAccuracy()
    
    with tf.GradientTape() as tape:
        # Forward pass
        enc_output = prot_cnn(inp, training=True)
        
        # Compute and aggregate loss
        loss += loss_function(targ, enc_output)
        
    # Compute average loss
    batch_loss = (loss / int(targ.shape[0]))
    
    # Get trainable variables, gradients, and update model parameters
    variables = prot_cnn.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    # Compute accuracy
    m.update_state(targ, enc_output)
    
    # batch_loss is a tf scaler value, we need to convert it to numpy object
    return batch_loss.numpy(), m.result().numpy()

def run_validation(X_val, Y_val, batch=1024):
    """
    This method computes loss and accuracy on validation or test data.
    
    Outputs:
        1. Average loss.
        2. Average accuracy.
    """
    
    # end_time is a variable to calculate the time needed to process data
    end_time = 0
    
    loss = 0
    m = tf.keras.metrics.CategoricalAccuracy()
    
    for i in range(0, len(X_val), batch):
        print("Working on batch {}".format(i//batch))
        
        # Pre-process sequences
        x_val = inp_df_to_np(X_val[i:i+batch])
        x_val = tf.keras.utils.to_categorical(x_val , amino_acids)

        # Pre-process labels
        y_val = tf.keras.utils.to_categorical(Y_val[i:i+batch] , unique_classes)
        
        start_time = time.time()
        validation_results = prot_cnn(x_val, training=False)
        end_time = (time.time() - start_time) + end_time

        loss += loss_function(y_val, validation_results)
        m.update_state(y_val, validation_results)
        
    return (loss / (i//batch + 1)).numpy(), m.result().numpy(), end_time
    
def main(args):
    global unique_classes, amino_acids, max_len, prot_cnn, optimizer, loss_object
    
    unique_classes = args.unique_classes
    amino_acids = args.amino_acids
    max_len = args.max_len
    
    print("Building model...")
    prot_cnn = ProtCNN(unique_classes)
    prot_cnn.build((None, max_len, amino_acids))
    
    print("Defining optimizer and loss function")
    optimizer_type = args.optimizer_type
    if (optimizer_type == "sgd"):
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.997, clipvalue=1.0)
    
    loss_object = tf.keras.losses.CategoricalCrossentropy()

    # Data directories
    train_dir = "/opt/ml/input/data/train"
    validation_dir = "/opt/ml/input/data/validation"
    test_dir = "/opt/ml/input/data/test"

    print("Defining checkpoint")
    # Define a checkpoint to save model while and after training
    checkpoint_dir = "/opt/ml/model"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(prot_cnn=prot_cnn)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=1)
    status = checkpoint.restore(manager.latest_checkpoint)
    
    # A dataframe to store history of training
    history = pd.DataFrame(columns=['train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy'])
    
    print("Starting training")
    
    # Get validation data
    validation_data = pd.read_csv(os.path.join(validation_dir, os.listdir(validation_dir)[0]))
    X_validation = validation_data["sequence"]
    Y_validation = validation_data["family_accession"]
    
    # Training data is splitted into 8 files
    partitions = [i for i in range(len(os.listdir(train_dir)))]
    
    EPOCHS = args.epochs
    
    # Training loop
    for epoch in range(EPOCHS):
        # Recored start time of an epoch
        epoch_start = time.time()
        
        train_loss = 0
        batch = args.batch_size
        train_acc = 0
        
        # total_batch keeps track the number of batchs accross the entire training data
        total_batch = 0
        
        # Loop over training partitions (files)
        for partition in partitions:
            print("Training on partition {}...".format(partition))
            
            test_data = pd.read_csv(os.path.join(train_dir, os.listdir(train_dir)[partition]))
            X_train = test_data["sequence"]
            Y_train = test_data["family_accession"]

            for i in range(0, len(X_train), batch):
                # Pre-process sequences
                x_train = inp_df_to_np(X_train[i:i+batch])
                x_train = tf.keras.utils.to_categorical(x_train , amino_acids)

                # Pre-process labels
                y_train = tf.keras.utils.to_categorical(Y_train[i:i+batch] , unique_classes)

                batch_loss, batch_acc = train_step(x_train, y_train)
                train_loss += batch_loss
                train_acc += batch_acc

                # Delete training objects to save memory
                del x_train
                del y_train

                if ((i//batch + total_batch) % 100 == 0):
                    print("Done with batch {} of epoch {}...".format(i//batch + total_batch, epoch+1))
                    print("Loss: {:.10f}, accuracy: {:.10f}".format(train_loss / (i//batch + total_batch + 1),
                                                                    train_acc / (i//batch + total_batch + 1)))
            # End of one partition
            
            total_batch += (i//batch + 1)
            
            # Perform a validation over a random batch from validation data
            random_batch = random.randint(0, len(X_validation)-batch)
            _, validation_acc, _ = run_validation(X_validation[random_batch:random_batch+batch],
                                                  Y_validation[random_batch:random_batch+batch])
            
            print("Random validation accuracy is {}".format(validation_acc))
            
        # End of an epoch
        
        # Save the progress so far
        checkpoint.save(file_prefix = checkpoint_prefix)

        train_loss = train_loss / total_batch
        train_acc = train_acc / total_batch

        print('Epoch {}, train loss {:.10f}, train accuracy {:.10f}'.format(epoch + 1, train_loss, train_acc))
        print('Time taken for an epoch is {} sec\n'.format(time.time() - epoch_start))

        # Perform validation
        print("Validating model...")
        validation_loss, validation_acc, validation_time = run_validation(X_validation, Y_validation)

        print("Validation loss is {}".format(validation_loss))
        print("Validation accuracy is {}".format(validation_acc))
        print("The model processes {} samples per second\n".format(round(len(X_validation)/validation_time, 4)))
        
        # Recored history
        history_dic = {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'validation_loss': validation_loss,
            'validation_accuracy': validation_acc
        }
        
        # Update dataframe
        history = history.append(history_dic, ignore_index=True)
        
    # End of training
    
    # Write history dataframe to S3
    csv_buffer = StringIO()
    history.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, 'history.csv').put(Body=csv_buffer.getvalue())
        
    # Perform testing...
    print("Testing model...")
    test_data = pd.read_csv(os.path.join(test_dir, os.listdir(test_dir)[0]))
    X_test = test_data["sequence"]
    Y_test = test_data["family_accession"]

    test_loss, test_acc, test_time = run_validation(X_test, Y_test)

    print("Test loss is {}".format(test_loss))
    print("Test accuracy is {}".format(test_acc))
    print("The model processes {} samples per second".format(round(len(X_test)/test_time, 4)))


if __name__ == '__main__':
    """
    This method is the entry point of the program
    """
    
    global bucket

    parser = argparse.ArgumentParser()

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--model_dir', type=str, required=True, help='The directory where the model will be stored.')
    parser.add_argument('--bucket', type=str, default="sagemaker-us-east-1-877465308896")
    
    # Training hyper-parameters
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--optimizer_type', type=str, default="adam")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    # Model hyper-parameters
    """
    The parameters are:
        1. unique_classes is the total number of classes to be predicted.
        2. max_len is the maximun length a sequence can have. Shorter sequences are padded with zeros.
        3. 25 amino_acids (indexed from 1) plus 1 to account for zero padding. So it defaults to 26 (i.e., 25 + 1). 
    """
    parser.add_argument('--unique_classes', type=int, default=17929)
    parser.add_argument('--max_len', type=int, default=2037)
    parser.add_argument('--amino_acids', type=int, default=26)
    
    args = parser.parse_args()
    bucket = args.bucket
    main(args)
