from tensorflow.keras import layers
import tensorflow as tf

class ProtCNN(tf.keras.Model):
    def __init__(self, out_prob):
        super(ProtCNN, self).__init__()
        self.out_prob = out_prob
        
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
        self.dense = layers.Dense(self.out_prob, name="dense",
                          kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))
        self.final_output = layers.Activation('softmax', name='output')
        
        
    def call(self, x):

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