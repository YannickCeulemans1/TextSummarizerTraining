import pickle
import numpy as np
import pandas as pd
import random

### Helper Functions

def gen_train_test_split_doc_level(Xy_doc_label, X, y, 
                                         test_ratio, folds=1, rand_seed=42):
    '''returns train doc labels, test doc labels, and train and test sets
    for features X and target Y'''
    
    random.seed(rand_seed)
    
    #index is doc label 
    total_docs = Xy_doc_label.max()
    train_docs_num = int(total_docs*(1-test_ratio))

    #for k >1, want to ensure different seeds
    rand_state_list = random.sample(range(2*folds), folds)
    
    #look through k folds
    train_test_set = []
    for state in rand_state_list:
        random.seed(state)
        #sample random training set and mask
        train_docs = random.sample(range(1, total_docs+1), train_docs_num)
        train_mask = np.array([x in train_docs for x in list(Xy_doc_label)])
        
        #use mask to define train and test sets
        X_train = X[train_mask]
        y_train = y[train_mask]
    
        X_test = X[~train_mask]
        y_test = y[~train_mask]
    
        Xy_doc_label_train = Xy_doc_label[train_mask]
        Xy_doc_label_test = Xy_doc_label[~train_mask]
        
        #assign all data to tuple for each pass
        data_pass = (Xy_doc_label_train, Xy_doc_label_test, X_train, X_test, y_train, y_test)
        #append results for ith fold to set 
        train_test_set.append(data_pass)
    
    #set answer tuples to final tuple as container
    train_test_set = tuple(train_test_set)

    return train_test_set


### Script

input_file = 'data/df_processed_label_formatted.pickle' 
folds = 1

output_file = 'data/train_test_embed_only_df_processed_label_formatted.pickle'

data_dict = pd.read_pickle(input_file)

#Specify model inputs: df, X, y, doc_labels
df = data_dict['df_original']
Xy_doc_label = data_dict['Xy_doc_label_array']
X = data_dict['df_X'].drop(['Sent_Number','Doc_Length'], axis=1).values
y = data_dict['y_array']

        
#train test split at document level

train_test_set = gen_train_test_split_doc_level(Xy_doc_label, X, y, test_ratio=0.2, folds=folds, rand_seed=42)

data_dict.update({'train_test_sets': train_test_set })

with open(output_file, 'wb') as handle:                                     
    pickle.dump(data_dict, handle)