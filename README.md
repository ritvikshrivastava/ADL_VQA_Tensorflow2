# Visual Question Answering in TensorFlow 2
Submission for Applied Deep Learning Course Project

Implemented by: Ritvik Shrivastava(rs3868), Saurabh Sharma(ss5569), Shivali Goel(sg3629)


Video Link: https://www.youtube.com/watch?v=srQ-C_Sa8PU

Github Link: https://github.com/saurabh1295/adl_vqa

For an end to end tutorial refer to the notebook: VQA_ADL.ipynb

To  run:

Download the dataset:

```wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

wget http://images.cocodataset.org/zips/train2014.zip - Unzip to ~/train2014

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip

unzip -a v2_Questions_Train_mscoco.zip

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip

unzip -a v2_Annotations_Train_mscoco.zip
```

The models are defined in the *model.py files.

To run the training script:

```train.py model_Type amount_of_data num_epochs dir/to/save/model```

model_type: pa - parallel coattention, aa - alternating coattention, iwi- IWIModel


The model is saved every 10 epochs.

get_d_copy.py - get_data(spatial = True) returns processed data. Spatial = True if using Coattention models. False for IWIModel.



Some sample answers and true answers are shown below:

![Sample Answers on a validation dataset](https://github.com/saurabh1295/adl_vqa/blob/master/images/screen.png)

![Sample Answers on a validation dataset](https://github.com/saurabh1295/adl_vqa/blob/master/images/screen2.png)

We will add an image demo soon!

Ref. paper: https://arxiv.org/abs/1606.00061
