from all_imports import *
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def get_data(spatial = True):

    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                              cache_subdir=os.path.abspath('.'),
                                              origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                              extract = True)
    annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'

    name_of_zip = 'train2014.zip'
    if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
      image_zip = tf.keras.utils.get_file(name_of_zip,
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                          extract = True)
      PATH = os.path.dirname(image_zip)+'/train2014/'
    else:

      PATH = os.path.abspath('.')+'/train2014/'

    print(PATH)
    """### Processing VQA Dataset"""

    import collections
    import operator
    # read the json file
    print("Reading annotation file...")
    annotation_file = 'v2_mscoco_train2014_annotations.json'
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # storing the captions and the image name in vectors
    all_answers = []
    all_answers_qids = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        #print(annot)
        ans_dic = collections.defaultdict(int)
        for each in annot['answers']:
          diffans = each['answer']
          if diffans in ans_dic:
            #print(each['answer_confidence'])
            if each['answer_confidence']=='yes':
              ans_dic[diffans]+=4
            if each['answer_confidence']=='maybe':
              ans_dic[diffans]+= 2
            if each['answer_confidence']=='no':
              ans_dic[diffans]+= 1
          else:
            if each['answer_confidence']=='yes':
              ans_dic[diffans]= 4
            if each['answer_confidence']=='maybe':
              ans_dic[diffans]= 2
            if each['answer_confidence']=='no':
              ans_dic[diffans]= 1
        #print(ans_dic)  
        most_fav = max(ans_dic.items(), key=operator.itemgetter(1))[0]
        #print(most_fav)
        caption = '<start> ' + most_fav + ' <end>' #each['answer']
        
        image_id = annot['image_id']
        question_id = annot['question_id']
        full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

        all_img_name_vector.append(full_coco_image_path)
        all_answers.append(caption)
        all_answers_qids.append(question_id)

    print("Done reading annotation file.")
    print("Reading Question file...")
    # read the json file
    question_file = 'v2_OpenEnded_mscoco_train2014_questions.json'
    with open(question_file, 'r') as f:
        questions = json.load(f)

    # storing the captions and the image name in vectors
    question_ids =[]
    all_questions = []
    all_img_name_vector_2 = []

    for annot in questions['questions']:
        caption = '<start> ' + annot['question'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

        all_img_name_vector_2.append(full_coco_image_path)
        all_questions.append(caption)
        question_ids.append(annot['question_id'])

    print(len(all_img_name_vector),len(all_answers), len(all_answers_qids))
    print(all_img_name_vector[10:15],all_answers[10:15], all_answers_qids[10:15])
    print(len(all_img_name_vector), len(all_questions) , len(question_ids))
    print(all_img_name_vector_2[10:15],all_questions[10:15], question_ids[10:15])

    # shuffling the captions and image_names together
    # setting a random state


    train_answers, train_questions, img_name_vector = shuffle(all_answers,all_questions,
                                              all_img_name_vector,
                                              random_state=1)

    print("Done pre processing Questions answers and images")
    print("Now preparing Image vectors...")
    # selecting the first 30000 captions from the shuffled set
    #num_examples = 3000
    #train_answers = train_answers[:num_examples]
    #train_questions = train_questions[:num_examples]
    #img_name_vector = img_name_vector[:num_examples]

    #print(img_name_vector[0],train_questions[0],train_answers[0])

    print("Length of image name vector ",len(img_name_vector),"Length of training questions ",len(train_questions)," Length of train answers ",len(train_answers))

    """### Getting Image Feature vector using VGG"""
    flag = False
    for path in img_name_vector:
        path_of_feature = path
        if spatial==False:
            if os.path.isfile(path_of_feature+"_dense.npy"):
                flag = True
        else:
            if os.path.isfile(path_of_feature+".npy"):
                flag = True
        break
    if flag == False:
        print("Using VGG Convolution base...")
        def load_image(image_path):
            img = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img, channels=3)
        #     224 x 224 for VGG 299x299 for Inception
            img = tf.image.resize(img, (224, 224))
            img = tf.keras.applications.inception_v3.preprocess_input(img)
            return img, image_path
        if(spatial == False):
            image_model = tf.keras.applications.VGG16(include_top=True,
                                                        weights='imagenet',input_shape = (224,224,3))
            new_input = image_model.input
            hidden_layer = image_model.layers[-2].output
        else:
            image_model = tf.keras.applications.VGG16(include_top=False,
                                                        weights='imagenet',input_shape = (224,224,3))
            new_input = image_model.input
            hidden_layer = image_model.layers[-1].output

        image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

        # getting the unique images
        encode_train = sorted(set(img_name_vector))

        # feel free to change the batch_size according to your system configuration
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        image_dataset = image_dataset.map(
          load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

        print("Converting..")
        for img, path in image_dataset:
          batch_features = image_features_extract_model(img)
          batch_features = tf.reshape(batch_features,
                                      (batch_features.shape[0], -1, batch_features.shape[1]))
          #print(batch_features.shape)

          for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            if spatial:
                sv_p = path_of_feature+".npy"
            else:
                sv_p = path_of_feature+"_dense.npy"
            np.save(sv_p, bf.numpy())
        
    print("Done getting image feature vectors")
    """### Creating Question Vectors"""
    print("Getting question vectors")
    # This will find the maximum length of any question in our dataset
    def calc_max_length(tensor):
        return max(len(t) for t in tensor)

    # choosing the top 10000 words from the vocabulary
    top_k = 10000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_questions)
    train_question_seqs = tokenizer.texts_to_sequences(train_questions)

    #new edit
    #print(tokenizer.word_index)
    ques_vocab = tokenizer.word_index
    #print(train_question_seqs)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # creating the tokenized vectors
    train_question_seqs = tokenizer.texts_to_sequences(train_questions)

    # padding each vector to the max_length of the captions
    # if the max_length parameter is not provided, pad_sequences calculates that automatically
    question_vector = tf.keras.preprocessing.sequence.pad_sequences(train_question_seqs, padding='post')
    #cap_vector

    # calculating the max_length
    # used to store the attention weights
    max_length = calc_max_length(train_question_seqs)
    print(max_length)

    #new edit
    max_q = max_length
    print("Done getting question feature vectors")
    """### Creating answer one hot vectors"""
    print("One hot encoding answer vectors...")
    # considering all answers to be part of ans vocab
    # define example
    data = train_answers
    values = array(data)
    print(values[:10])

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    #print(integer_encoded)
    #new edit
    ans_vocab = {l: i for i, l in enumerate(label_encoder.classes_)}
    print("Length of answer vocab",len(ans_vocab))

    # binary encode
    #onehot_encoder = OneHotEncoder(sparse=False)
    #integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    #onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #print(onehot_encoded[0],len(onehot_encoded))

    #answer_vector = onehot_encoded

    #new edit
    #len_ans_vocab = len(onehot_encoded[0])

    #print(answer_vector)
    #print(len(question_vector[0]), len(answer_vector[0]))

    """### TRAIN - TEST SPLIT"""

    img_name_train, img_name_val, question_train, question_val,answer_train, answer_val  = train_test_split(img_name_vector,
                                                                        question_vector,
                                                                        integer_encoded,
                                                                        test_size=0.1,
                                                                        random_state=0)

    print(len(img_name_train), len(img_name_val), len(question_train), len(question_val),len(answer_train), len(answer_val))

    """### Almost done with data processing!!!"""

    # feel free to change these parameters according to your system's configuration

    BATCH_SIZE = 64 #2 #64
    BUFFER_SIZE = 1000 #1000
    # embedding_dim = 256
    # units = 512
    # vocab_size = len(tokenizer.word_index) + 1
    num_steps = len(img_name_train) // BATCH_SIZE
    # shape of the vector extracted from InceptionV3 is (64, 2048)
    # these two variables represent that
    features_shape = 512
    attention_features_shape = 49

    # loading the numpy files
    def map_func(img_name, cap,ans):
      img_tensor = np.load(img_name.decode('utf-8')+'.npy')
      return img_tensor, cap,ans

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, question_train.astype(np.float32), answer_train.astype(np.float32)))

    # using map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2, item3: tf.numpy_function(map_func, [item1, item2, item3], [tf.float32, tf.float32, tf.float32]),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # shuffling and batching
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder = True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((img_name_val, question_val.astype(np.float32), answer_val.astype(np.float32)))

    # using map to load the numpy files in parallel
    test_dataset = test_dataset.map(lambda item1, item2, item3: tf.numpy_function(
              map_func, [item1, item2, item3], [tf.float32, tf.float32, tf.float32]),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # shuffling and batching
    test_dataset = test_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


    return dataset,test_dataset,ques_vocab,ans_vocab,max_q,label_encoder,tokenizer
