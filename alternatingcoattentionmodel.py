"""
Author: Ritvik Shrivastava

"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, Embedding, LSTM, Activation,ZeroPadding1D,Conv1D

class AlternatingCoattentionModel(tf.keras.Model):
  def __init__(self,ans_vocab,max_q,ques_vocab):
    super(AlternatingCoattentionModel, self).__init__(name='AlternatingCoattentionModel')
    self.ans_vocab = ans_vocab
    self.max_q = max_q
    self.ques_vocab = ques_vocab
    
    
    self.ip_dense = Dense(256, activation=None, input_shape=(512,)) 
    num_words = len(ques_vocab)+2
    self.word_level_feats = Embedding(input_dim = len(ques_vocab)+2,output_dim = 256)
    self.lstm_layer = LSTM(256,return_sequences=True,input_shape=(None,max_q,256)) 
    self.dropout_layer = Dropout(0.5)
    self.tan_layer = Activation('tanh')
    self.phrase_level_unigram = Conv1D(256,kernel_size=256,strides=256) 
    self.phrase_level_bigram = Conv1D(256,kernel_size=2*256,strides=256,padding='same') 
    self.phrase_level_trigram = Conv1D(256,kernel_size=3*256,strides=256,padding='same') 
    self.dense_image = Dense(256, activation=None, input_shape=(256,))
    self.dense_text = Dense(256, activation=None, input_shape=(256,))
    self.image_attention = Dense(1, activation='softmax', input_shape=(256,))
    self.text_attention = Dense(1, activation='softmax', input_shape=(256,)) 
    self.dense_word_level = Dense(256, activation=None, input_shape=(256,)) 
    self.dense_phrase_level = Dense(256, activation=None, input_shape=(2*256,)) 
    self.dense_sent_level = Dense(256, activation=None, input_shape=(2*256,)) 
    self.dense_final = Dense(len(ans_vocab), activation=None, input_shape=(256,))
    
    
  def affinity(self,image_feat,text_feat,g,prev_att):
    V_ = self.dense_image(image_feat)
    Q_ = self.dense_text(text_feat)
    
    if g==0:
      temp1 = self.tan_layer(Q_)
      H_text = self.dropout_layer(temp1) 
      return H_text
    
    elif g==1:
      g = self.dense_text(prev_att)   
      g = tf.expand_dims(g,1)
      temp = V_ + g
      temp = self.tan_layer(temp)
      H_img = self.dropout_layer(temp)
      return H_img
      
    elif g==2:
      g = self.dense_image(prev_att)
      g = tf.expand_dims(g,1)
      temp = Q_ + g
      temp = self.tan_layer(temp)
      H_text = self.dropout_layer(temp)
      return H_text
    
  
  def attention_ques(self,text_feat,H_text):
    temp = self.text_attention(H_text)
    return tf.reduce_sum(temp * text_feat,1) 
  
  
  def attention_img(self,image_feat,H_img):
    temp = self.image_attention(H_img)
    return tf.reduce_sum(temp * image_feat,1)
    
  def call(self,image_feat,question_encoding):
    # Processing the image
    image_feat = self.ip_dense(image_feat) 

    # Text fetaures 

    # Text: Word level
    word_feat = self.word_level_feats(question_encoding) 

    # Text: Phrase level
    word_feat_ = tf.reshape(word_feat,[word_feat.shape[0], 1, -1])
    word_feat_= tf.transpose(word_feat_, perm=[0,2,1]) 
    uni_feat = self.phrase_level_unigram(word_feat_)
    uni_feat = tf.expand_dims(uni_feat,-1) 
    bi_feat = self.phrase_level_bigram(word_feat_) 
    bi_feat = tf.expand_dims(bi_feat,-1)
    tri_feat = self.phrase_level_trigram(word_feat_)
    tri_feat = tf.expand_dims(tri_feat,-1)
    all_feat = tf.concat([uni_feat, bi_feat, tri_feat],-1)
    phrase_feat = tf.reduce_max(all_feat,-1) 

    # Text: Sentence level
    sent_feat = self.lstm_layer(phrase_feat) 

  	#Apply attention to features at all three levels

    # Applying attention on word level features
    word_H_text = self.affinity(image_feat,word_feat,0,0)
    word_text_attention = self.attention_ques(word_feat,word_H_text)
    word_H_img = self.affinity(image_feat,word_feat,1,word_text_attention)
    word_img_attention = self.attention_img(image_feat,word_H_img)
    word_H_text = self.affinity(image_feat,word_feat,2,word_img_attention)
    word_text_attention = self.attention_ques(word_feat,word_H_text)

    word_level_attention = word_img_attention + word_text_attention
    word_pred = self.dropout_layer(self.tan_layer(self.dense_word_level(word_level_attention)))

    # Applying attention on phrase level features
    phrase_H_text = self.affinity(image_feat,phrase_feat,0,0)
    phrase_text_attention = self.attention_ques(phrase_feat,phrase_H_text)
    phrase_H_img = self.affinity(image_feat,phrase_feat,1,phrase_text_attention)
    phrase_img_attention = self.attention_img(image_feat,phrase_H_img)
    phrase_H_text = self.affinity(image_feat,phrase_feat,2,phrase_img_attention)
    phrase_text_attention = self.attention_ques(phrase_feat,phrase_H_text)

    phrase_level_attention = tf.concat([phrase_img_attention + phrase_text_attention, word_pred],-1) 
    phrase_pred = self.dropout_layer(self.tan_layer(self.dense_phrase_level(phrase_level_attention)))

    # Applying attention on sentence level features
    sent_H_text = self.affinity(image_feat,sent_feat,0,0)
    sent_text_attention = self.attention_ques(sent_feat,sent_H_text)
    sent_H_img = self.affinity(image_feat,sent_feat,1,sent_text_attention)
    sent_img_attention = self.attention_img(image_feat,sent_H_img)
    sent_H_text = self.affinity(image_feat,sent_feat,2,sent_img_attention)
    sent_text_attention = self.attention_ques(sent_feat,sent_H_text)

    sentence_level_attention = tf.concat([sent_img_attention + sent_text_attention, phrase_pred],-1) 
    sent_pred = self.dropout_layer(self.tan_layer(self.dense_sent_level(sentence_level_attention)))


    return self.dense_final(sent_pred)
