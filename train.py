from  all_imports import *
from func_defs import train_step,test_step
from parallelcoattentionmodel import ParallelCoattentionModel
from iwimodel import IWIModel
from alternatingcoattentionmodel import AlternatingCoattentionModel
import argparse

from get_d_copy import get_data

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('model', help="Model to use: options ParallelCoattention use pa for Alternating Coattention use aa")
  parser.add_argument('num_samples', help="Number of samples to train on. Use all to train on the whole dataset")
  parser.add_argument('num_epochs', help="Number of epochs")
  parser.add_argument('save_path',help="Path where model weights will be stored")
  args = parser.parse_args()
  model_choice = str(args.model)
  spatial = True
  if(model_choice == "iwi"):
      spatial = False

    #return dataset,test_dataset,ques_vocab,ans_vocab,max_q,label_encoder,tokenizer
  dataset,test_dataset,ques_vocab,ans_vocab,max_q,label_encoder,tokenizer = get_data(spatial)
  if(model_choice=="pa"):
    model = ParallelCoattentionModel(ans_vocab,max_q,ques_vocab)
  elif(model_choice=="aa"):
    model = AlternatingCoattentionModel(ans_vocab,max_q,ques_vocab)
  else:
    model = IWIModel('vgg',100,256,128,len(ques_vocab),len(ans_vocab))

  EPOCHS = int(args.num_epochs)
  train_loss =[]
  test_loss=[]
  train_acc=[]
  test_acc=[]
  if(os.path.isdir(str(args.save_path))):
    save_prefix = os.path.join(str(args.save_path),str(args.model))
  else:
    print("The directory you provided does not exist")
    exit()
  model.save_weights(str(args.save_path+"/"+str(-1)+".h5")
  for epoch in range(EPOCHS):
    for (batch, (img_tensor, question, answer)) in enumerate(dataset):
        train_step(img_tensor, question, answer ,model)
  
    for (batch, (img_tensor, question, answer)) in enumerate(test_dataset):
        test_step(img_tensor, question, answer,model)
  
    template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}, Test loss: {:.4f}, Test accuracy: {:.2f}'
    train_loss.append(train_loss_metric.result())
    test_loss.append(test_loss_metric.result())
    train_acc.append(train_accuracy_metric.result() * 100)
    test_acc.append(test_accuracy_metric.result() * 100)
    print (template.format(epoch +1, 
                         train_loss_metric.result(), 
                         train_accuracy_metric.result() * 100, 
                        test_loss_metric.result(), 
                         test_accuracy_metric.result() * 100))
  if epoch  % 10 == 0:
    model.save_weights(str(args.save_path+"/"+str(epoch)+".h5"))
