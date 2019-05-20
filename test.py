from all_imports import *
from get_d_copy import get_data
from parallelcoattentionmodel512 import ParallelCoattentionModel
def test_vqa(num = 10):
    ctr = 0    
    for (batch, (img_tensor, question, answer,p_answers,img_name)) in enumerate(test_dataset):
            
            p = model(img_tensor,question)
            p = tf.math.top_k(p,k=2)
            p_indices = p.indices.numpy()
            questions = question.numpy()
            t_image_names = img_name.numpy()
            p_answers = p_answers.numpy()
            for i in range(len(t_image_names)):
                image = mpimg.imread(t_image_names[i].decode('utf-8'))
                plt.imshow(image)
                plt.show()
                print("Question: ",tokenizer.sequences_to_texts([questions[i]]))
                print("Predicted answers: ",label_encoder.inverse_transform(p_indices[i]))
                poss = p_answers[i]
                poss = poss[poss>=0]
                print("Correct answers:")
                for opt in poss:
                    print(label_encoder.inverse_transform([int(opt)]))
                ctr += 1
                if ctr==10:
                    break
            if ctr == 10:
                break
    #         print(label_encoder.inverse_transform(p_answers[np.argwhere(p_answers>=0)]))
            break

if __name__ == "__main__":
    dataset,test_dataset,ques_vocab,ans_vocab,max_q,label_encoder,tokenizer = get_data()
    model = ParallelCoattentionModel(ans_vocab,max_q,ques_vocab)
    for (batch, (img_tensor, question, answer)) in enumerate(test_dataset): 
            p = model(img_tensor,question)
            break

    model.load_weights("models/parallel_coattention_512/epoch_40.h5")
    ctr = 0    
    for (batch, (img_tensor, question, answer)) in enumerate(test_dataset):
            
            p = model(img_tensor,question)
            p = tf.math.top_k(p,k=2)
            p_indices = p.indices.numpy()
            questions = question.numpy()
            answer = answer.numpy()
           # t_image_names = img_name.numpy()
           # p_answers = p_answers.numpy()
            for i in range(len(questions)):
               # image = mpimg.imread(t_image_names[i].decode('utf-8'))
               # plt.imshow(image)
               # plt.show()
                print("Question: ",tokenizer.sequences_to_texts([questions[i]]))
                print("Predicted answers: ",label_encoder.inverse_transform(p_indices[i]))
               # poss = p_answers[i]
               # poss = poss[poss>=0]
                print("Correct answers:")
            #    for opt in poss:
                print(label_encoder.inverse_transform([int(answer[i])]))
                ctr += 1
                if ctr==10:
                    break
            if ctr == 10:
                break
    #         print(label_encoder.inverse_transform(p_answers[np.argwhere(p_answers>=0)]))
            break    
