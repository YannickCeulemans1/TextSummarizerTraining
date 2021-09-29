import numpy as np
import pandas as pd
import pickle
from datetime import datetime as dt

t1 = dt.now()
print(t1)
    
input_file = 'data/df_processed_label.pickle'
output_file = 'data/df_processed_label_formatted.pickle'
    
df = pd.read_pickle(input_file)

#define features and labels as separate series
s_embed_text = df.text_embedding
s_y_labels= df.labels

#label docs
s_doc_label = pd.Series(range(df.shape[0]), name = 'doc_label')

#calculate doc mean
s_doc_mean = s_embed_text.apply(lambda x: x.mean(axis=0).reshape(1,-1))
    
#calculate doc sent length
s_doc_length = s_embed_text.apply(lambda x: x.shape[0])


#create values for each sentence in doc 
X_doc_label_list =[]
X_doc_mean_list = []
X_doc_length_list = []
X_sent_num_list = []

for j in range(len(df)):
    X_doc_label = s_doc_label[j]
    X_doc_mean = s_doc_mean[j]
    X_doc_length = s_doc_length[j]
    X_text = s_embed_text [j]
    n = X_text.shape[0]

    X_doc_label_fixed = X_doc_label
    X_doc_mean_fixed = X_doc_mean
    X_doc_length_fixed = X_doc_length 
    sent_num = []
    for i in range(n-1): 
        X_doc_label = np.vstack((X_doc_label, X_doc_label_fixed )) 
        X_doc_mean = np.vstack((X_doc_mean, X_doc_mean_fixed )) 
        X_doc_length = np.vstack((X_doc_length, X_doc_length_fixed )) 
        sent_num.append(i)
    sent_num.append(n-1)
    
    X_doc_label_list.append(X_doc_label)
    X_doc_mean_list.append(X_doc_mean)
    X_doc_length_list.append(X_doc_length)
    X_sent_num_list.append(np.array(sent_num).reshape(-1,1))
    
#from list to pandas series
s_doc_label = pd.Series(X_doc_label_list)
s_doc_mean = pd.Series(X_doc_mean_list)
s_doc_length = pd.Series(X_doc_length_list)
s_sent_num = pd.Series(X_sent_num_list)

#concatenate documents with rows = sentences
  #intialize
Xy_doc_label = s_doc_label.values[0]
X = np.hstack((s_embed_text[0], s_doc_mean[0], s_sent_num[0], s_doc_length[0]))
y= s_y_labels[0].reshape(-1,1)
  #recursive population
f = np.vectorize(lambda x: x if type(x) == np.ndarray else np.array([[x]]))  
for j in range(1, len(df)):
    Xy_doc_label_new = s_doc_label.values[j]
    
    X_text_new = s_embed_text[j]
    X_sent_num_new = s_sent_num[j]
    X_doc_mean_new = s_doc_mean[j]
    X_doc_length_new = f(s_doc_length[j])
    y_new = s_y_labels[j].reshape(-1,1)
    
    X_new = np.hstack((X_text_new, X_doc_mean_new, X_sent_num_new, X_doc_length_new))
    
    X = np.vstack((X, X_new))
    y = np.vstack((y, y_new))           
    
    Xy_doc_label = np.vstack((Xy_doc_label, Xy_doc_label_new))
        
#wrap X in dataframe with lables
labels_text_embedding = ['Sent_BERT_D_' + str(j) for j in range(768)]
labels_doc_mean = ['Doc_BERT_D_' + str(j) for j in range(768)]
other_labels = ['Sent_Number', 'Doc_Length']
col_names = labels_text_embedding + labels_doc_mean + other_labels

df_X = pd.DataFrame(X, columns = col_names)
    
data_dict = {'df_original': df, 'Xy_doc_label_array': Xy_doc_label, 
              'df_X': df_X, 'y_array': y}
    
with open(output_file, 'wb') as handle:                                     
    pickle.dump(data_dict, handle)
    
t2 = dt.now()

print(t2)
print(t2-t1)