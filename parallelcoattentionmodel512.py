import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Dropout, Embedding, LSTM, Activation,ZeroPadding1D,Conv1D
from tensorflow.keras import Model

class ParallelCoattentionModel(tf.keras.Model):
  def __init__(self,ans_vocab,max_q,ques_vocab):
    super(ParallelCoattentionModel, self).__init__(name='ParallelCoattentionModel')
    self.ans_vocab = ans_vocab
    self.max_q = max_q
    self.ques_vocab = ques_vocab
    
    self.ip_dense = Dense(512, activation=None, input_shape=(512,)) 
    num_words = len(ques_vocab)+2
    self.word_level_feats = Embedding(input_dim = len(ques_vocab)+2,output_dim = 512)
    self.lstm_layer = LSTM(512,return_sequences=True,input_shape=(None,max_q,512)) 
    self.dropout_layer = Dropout(0.5)
    self.tan_layer = Activation('tanh')
    self.phrase_level_unigram = Conv1D(512,kernel_size=512,strides=512) 
    self.phrase_level_bigram = Conv1D(512,kernel_size=2*512,strides=512,padding='same') 
    self.phrase_level_trigram = Conv1D(512,kernel_size=3*512,strides=512,padding='same') 
    self.basic_dense_layer = Dense(512, activation=None, input_shape=(512,)) 
    self.dense_image = Dense(512, activation=None, input_shape=(512,))
    self.dense_text = Dense(512, activation=None, input_shape=(512,))
    self.image_attention = Dense(1, activation='softmax', input_shape=(512,))
    self.text_attention = Dense(1, activation='softmax', input_shape=(512,)) 
    self.dense_word_level = Dense(512, activation=None, input_shape=(512,)) 
    self.dense_phrase_level = Dense(512, activation=None, input_shape=(2*512,))
    self.dense_sent_level = Dense(512, activation=None, input_shape=(2*512,)) 
    self.dense_final = Dense(len(ans_vocab), activation=None, input_shape=(512,))
    
	
  def affinity(self,image_feat,text_feat): 
    temp_C = self.basic_dense_layer(image_feat)  
    temp_C =  tf.transpose(temp_C, perm=[0,2,1]) 
    C_ = tf.matmul(text_feat, temp_C) 
    C_ = self.tan_layer(C_)
    C_ = self.dropout_layer(C_) #Refer eqt (3) section 3.3 
    V_ = self.dense_image(image_feat)
    Q_ = self.dense_text(text_feat)
    QC_ = tf.matmul(tf.transpose(Q_, perm=[0,2,1]),C_) 
    QC_ = tf.transpose(QC_, perm =[0,2,1]) 
    temp1 = V_ + QC_
    temp1 = self.tan_layer(temp1)
    H_img = self.dropout_layer(temp1) #Refer eqt (4) section 3.3 
    VC_ = tf.matmul(tf.transpose(V_, perm=[0,2,1]),tf.transpose(C_,perm=[0,2,1])) 
    VC_ = tf.transpose(VC_, perm =[0,2,1]) 
    temp2 = Q_ + VC_
    temp2 = self.tan_layer(temp2)
    H_text = self.dropout_layer(temp2)
    return H_img, H_text
  
  def parallel_attention(self,image_feat,text_feat,H_img,H_text): 
    a_img = self.image_attention(H_img)
    a_text = self.text_attention(H_text)
    return tf.reduce_sum(a_img * image_feat,1), tf.reduce_sum(a_text * text_feat,1) #Refer eqt (5) section 3.3

    
  def call(self,image_feat,question_encoding): 
    # Image features
    image_feat = self.ip_dense(image_feat) #check input shape (512,) ???
    
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
    sent_feat = self.lstm_layer(phrase_feat) #temp1 #old lstm_
    
    #Applying attention
    # All variable names correspond to Fig.3(b)in paper. 
    
    # Attention on word level features
    w_H_img,w_H_text = self.affinity(image_feat,word_feat)
    v_w,q_w = self.parallel_attention(image_feat,word_feat,w_H_img,w_H_text)
    h_w = v_w + q_w
    h_w_ = self.dropout_layer(self.tan_layer(self.dense_word_level(h_w)))
    
    # Attention on phrase level features
    p_H_img,p_H_text = self.affinity(image_feat,phrase_feat)
    v_p,q_p = self.parallel_attention(image_feat,phrase_feat,p_H_img,p_H_text)
    h_p = tf.concat([v_p + q_p, h_w_],-1) 
    h_p_ = self.dropout_layer(self.tan_layer(self.dense_phrase_level(h_p)))
    
    # Attention on sentence level features
    s_H_img,s_H_text = self.affinity(image_feat,sent_feat)
    v_s,q_s = self.parallel_attention(image_feat,sent_feat,s_H_img,s_H_text)
    h_s = tf.concat([v_s + q_s, h_p_],-1) 
    h_s_ = self.dropout_layer(self.tan_layer(self.dense_sent_level(h_s)))

    return self.dense_final(h_s_)
