'''
Image with Words with Image.
This model uses the image twice. First it treats the image as a word in the question vector. 
But this will cause the image to be weighted less as it is treated ~ 1/20th of the question vector
We therefore concatenate the image features from the CNN to the output of the LSTM again to extract more information.
'''
import tensorflow as tf
class IWIModel(tf.keras.Model):
    
    def __init__(self,ques_vocab_size,ans_vocab_size):
        super(IWIModel, self).__init__()
        #self.ip_shape = ip_shape
        self.rnn_size = 256
        self.embedding_size = 64
        self.flatten = tf.keras.layers.Flatten()
        self.image_dense = tf.keras.layers.Dense(self.embedding_size,activation='relu')
        self.q_embedding = tf.keras.layers.Embedding(input_dim = ques_vocab_size+2,output_dim = self.embedding_size)
        self.gru_layer = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.rnn_size,return_state = True))
        #self.gru_layer = tf.keras.layers.GRU(self.rnn_size,return_state=True,return_sequences=False)
        self.d_after_gru = tf.keras.layers.Dense(128,activation = 'relu')
      #  self.d_after_gru_2 = tf.keras.layers.Dense(256,activation = 'relu')
        self.softmax = tf.keras.layers.Dense(ans_vocab_size,activation='softmax')
        
    def call(self,input_image,input_q):
        image_feats_f = self.flatten(input_image)
        image_feats = self.image_dense(image_feats_f)
        image_feats = tf.expand_dims(image_feats,axis =1)
        input_q = self.q_embedding(input_q)
        rnn_input = tf.concat([input_q,image_feats],axis = 1)
        output1,output2,state = self.gru_layer(rnn_input)
        softmax_input = tf.concat([output1,output2,image_feats_f],axis = 1)
        softmax_input = self.d_after_gru(softmax_input)
       # softmax_input = self.d_after_gru_2(softmax_input)
        output = self.softmax(softmax_input)
        return output

    def init_state(self, batch_size):
        return tf.zeros((batch_size, rnn_size))

