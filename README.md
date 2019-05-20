# Visual Question Answering
Submission for Applied Deep Learning VQA

To run:
Download the dataset:
Annotations: http://images.cocodataset.org/annotations/annotations_trainval2014.zip
Images: http://images.cocodataset.org/zips/train2014.zip - Unzip to ~/train2014
!wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
! unzip -a v2_Questions_Train_mscoco.zip
!wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
! unzip -a v2_Annotations_Train_mscoco.zip

The models are defined in the *model.py files.

To run the training script:
train.py pa all 10 dir/to/save/model
pa - parallel coattention
aa - alternating coattention


The model is saved every 10 epochs.

get_d_copy.py - get_data(spatial = True) returns processed data. Spatial = True if using Coattention models. False for IWIModel.



Some sample answers and true answers are shown below:

![Sample Answers on a validation dataset](https://github.com/saurabh1295/adl_vqa/blob/master/images/screen.png)

![Sample Answers on a validation dataset](https://github.com/saurabh1295/adl_vqa/blob/master/images/screen2.png)

We will add an image demo soon!

Ref. paper: https://arxiv.org/abs/1606.00061


Implemented by: Ritvik Shrivastava, Saurabh Sharma, Shivali Goel
