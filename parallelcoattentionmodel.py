
import tensorflow as tf
from all_imports import *
class ParallelCoattentionModel(tf.keras.Model):
  def __init__(self,ans_vocab,max_q,ques_vocab):
    super(ParallelCoattentionModel, self).__init__(name='ParallelCoattentionModel')
    self.ans_vocab = ans_vocab
    self.max_q = max_q
    self.ques_vocab = ques_vocab
    
    self.ip_dense = Dense(256, activation=None, input_shape=(512,)) 
    num_words = len(ques_vocab)+2
    self.word_feat = Embedding(input_dim = len(ques_vocab)+2,output_dim = 256)
    self.lstm_ = LSTM(256,return_sequences=True,input_shape=(None,max_q,256)) # or num words check ...
    self.dropout_layer = Dropout(0.5)
    self.tan_layer = Activation('tanh')
    self.unigram = Conv1D(256,kernel_size=256,strides=256)
    self.bigram = Conv1D(256,kernel_size=2*256,strides=256,padding='same')
    self.trigram = Conv1D(256,kernel_size=3*256,strides=256,padding='same')
    self.dense_inter = Dense(256, activation=None, input_shape=(256,)) 
    self.dense_image = Dense(256, activation=None, input_shape=(256,))
    self.dense_text = Dense(256, activation=None, input_shape=(256,))
    self.dense_att_image = Dense(1, activation='softmax', input_shape=(256,)) 
    self.dense_att_text = Dense(1, activation='softmax', input_shape=(256,))
    self.dense_word = Dense(256, activation=None, input_shape=(256,))
    self.dense_phrase = Dense(256, activation=None, input_shape=(2*256,)) 
    self.dense_sent = Dense(256, activation=None, input_shape=(2*256,))
    self.dense_final = Dense(len(ans_vocab), activation=None, input_shape=(256,))
    
	
  def affinity(self,image_feat,text_feat): #STATUS CHECKED
    temp_inter = self.dense_inter(image_feat)
    temp_inter =  tf.transpose(temp_inter, perm=[0,2,1]) 
    inter = tf.matmul(text_feat, temp_inter) 
    inter = self.tan_layer(inter)
    inter = self.dropout_layer(inter)
    V_ = self.dense_image(image_feat)
    Q_ = self.dense_text(text_feat)
    temp1 = tf.matmul(tf.transpose(Q_, perm=[0,2,1]),inter) 
    temp1 = tf.transpose(temp1, perm =[0,2,1]) 
    temp2 = V_ + temp1
    temp2 = self.tan_layer(temp2)
    H_img = self.dropout_layer(temp2)
    temp1 = tf.matmul(tf.transpose(V_, perm=[0,2,1]),tf.transpose(inter,perm=[0,2,1])) 
    temp1 = tf.transpose(temp1, perm =[0,2,1]) 
    temp2 = Q_ + temp1
    temp2 = self.tan_layer(temp2)
    H_text = self.dropout_layer(temp2)
    return H_img, H_text
  
  def attention_mask(self,image_feat,text_feat,H_img,H_text):
    temp1 = self.dense_att_image(H_img)
    temp2 = self.dense_att_text(H_text)
    return tf.reduce_sum(temp1 * image_feat,1), tf.reduce_sum(temp2 * text_feat,1) 

    
  def call(self,image_feat,question_encoding): #STATUS 1 CHECK REQUIRED
    # Processing the image
    image_feat = self.ip_dense(image_feat) #check input shape (512,) ???
    
    # Processing the text
    
    # Word level features
    w_feat = self.word_feat(question_encoding)
    
    # Phrase level features
    encode_ = tf.reshape(w_feat,[w_feat.shape[0], 1, -1])
    encode_= tf.transpose(encode_, perm=[0,2,1]) 
    u_feat = self.unigram(encode_)
    u_feat = tf.expand_dims(u_feat,-1) 
    bi_feat = self.bigram(encode_) 
    bi_feat = tf.expand_dims(bi_feat,-1)
    tri_feat = self.trigram(encode_)
    tri_feat = tf.expand_dims(tri_feat,-1)
    all_feat = tf.concat([u_feat, bi_feat, tri_feat],-1)
    p_feat = tf.reduce_max(all_feat,-1) 
    
    # Sentence level Features
    #temp1 = tf.transpose(p_feat, perm=[0,2,1]) 
    s_feat = self.lstm_(p_feat) #temp1
    
    
    #Apply attention to features at all three levels
    
    # Applying attention on word level features
    word_H_img,word_H_text = self.affinity(image_feat,w_feat)
    word_vis_attention,word_text_attention = self.attention_mask(image_feat,w_feat,word_H_img,word_H_text)
    word_level_attention = word_vis_attention + word_text_attention
    word_pred = self.dropout_layer(self.tan_layer(self.dense_word(word_level_attention)))
    
    # Applying attention on phrase level features
    phrase_H_img,phrase_H_text = self.affinity(image_feat,p_feat)
    phrase_vis_attention,phrase_text_attention = self.attention_mask(image_feat,p_feat,phrase_H_img,phrase_H_text)
    phrase_level_attention = tf.concat([phrase_vis_attention + phrase_text_attention, word_pred],-1) #torch.cat((phrase_vis_attention + phrase_text_attention, word_pred), -1)
    phrase_pred = self.dropout_layer(self.tan_layer(self.dense_phrase(phrase_level_attention)))
    
    # Applying attention on sentence level features
    sent_H_img,sent_H_text = self.affinity(image_feat,s_feat)
    sent_vis_attention,sent_text_attention = self.attention_mask(image_feat,s_feat,sent_H_img,sent_H_text)
    sentence_level_attention = tf.concat([sent_vis_attention + sent_text_attention, phrase_pred],-1) #torch.cat((sent_vis_attention + sent_text_attention, phrase_pred), -1)
    sent_pred = self.dropout_layer(self.tan_layer(self.dense_sent(sentence_level_attention)))

    return self.dense_final(sent_pred)
