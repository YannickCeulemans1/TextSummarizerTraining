import pickle
import numpy as np
import pandas as pd
from datetime import datetime as dt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from datasets import load_metric 

### Functions

def return_greater_than_min_num(arr, thresh=0.5, min_num=1, fix_num_flag=False, fix_num=3):
    print(arr) 
    if fix_num_flag == True:
        idx = np.argsort(arr)[-fix_num:]
        
    else:
        idx_prelim = np.where(arr>= thresh)
    
        if idx_prelim[0].shape[0] <= min_num:
            idx = np.argsort(arr)[-min_num:]
        else:
            idx = idx_prelim
            idx = idx[0]

    return sorted(idx)

def return_df_pred_summaries( Xy_doc_label, y_pred, df_text, thresh, min_num,
                             return_all=False, fix_num_flag=False, fix_num=3):
    
    df_label_pred = pd.DataFrame({'doc_label': Xy_doc_label.flatten(),
                                                 'y_pred': y_pred.flatten()}) 
    df_label_pred = df_label_pred.groupby('doc_label').agg(list) 

    df_label_pred = df_label_pred.applymap(lambda x: np.array(x))

    f = lambda arr: return_greater_than_min_num(arr, thresh=thresh, 
                                    min_num=min_num,fix_num_flag = fix_num_flag, 
                                                            fix_num=fix_num)

    df_label_pred = df_label_pred.applymap(f) 

    #Return predicted summary
    df_doc = df_text[df_label_pred.index]
    
    
    pred_summaries = [np.array(df_doc.iloc[j])
                               [df_label_pred.iloc[j][0]].tolist()                      #???
                                          for j in range(len(df_label_pred))]

    pred_summaries = [summ_list if type(summ_list) == str else 
                      ' '.join(summ_list) for summ_list in pred_summaries]
    
    if return_all == True:
        answer = df_label_pred.values, df_label_pred.index, pred_summaries
    else:
        answer = pred_summaries
    
    return answer
    
### Script

input_file = 'data/train_test_embed_only_df_processed_label_formatted.pickle'
metric = load_metric("rouge")
output_file = 'data/results/logisticRegression_embed_only.pickle'
model_file = 'data/results/logisticRegression_model.pickle'

data_dict = pd.read_pickle(input_file)

#Specify model inputs: df, X, y, doc_labels
df = data_dict['df_original']
train_test_set = data_dict['train_test_sets']
#Specify train-test_data for validation        
Xy_doc_label_train = train_test_set[0][0]
Xy_doc_label_test = train_test_set[0][1]
X_train = train_test_set[0][2]
X_test = train_test_set[0][3]
y_train = train_test_set[0][4]
y_test = train_test_set[0][5]

#Define Model
model = LogisticRegression(random_state=42,max_iter=10000)
#Fit model
model.fit(X_train,y_train.ravel())
#Save model 
with open(model_file, 'wb') as handle:                                     
    pickle.dump(model, handle)
#Predict Model
y_pred = model.predict_proba(X_test)

    
#Convert to binary predictions
y_pred_bin = (y_pred >=0.5)*1

cm = confusion_matrix(y_test, y_pred_bin[:,1], labels=[0,1])

#Return predicted summaries
idx, doc_index, pred_summaries = return_df_pred_summaries(Xy_doc_label_test, y_pred[:,1], df.text_clean, thresh=0.5, min_num=1, return_all = True, fix_num_flag = False, fix_num=3)          

#Match with gold summaries
df_gold = df.sum_clean[doc_index]
gold_summaries = [' '.join(df_gold .iloc[j]) for j in range(len(pred_summaries))]
summaries_comp = tuple(zip(pred_summaries, gold_summaries))

# scores = calc_rouge_scores(pred_summaries, gold_summaries, keys=['rouge1', 'rougeL'], use_stemmer=True)
metric.add_batch(predictions=pred_summaries,references=gold_summaries)

scores = metric.compute()
#print(scores)

results_dict ={'conf_matrix': cm, 'summaries_comp': summaries_comp,
               'sent_index_number': idx, 'Rouge': scores}

with open(output_file, 'wb') as handle:                                     
    pickle.dump(results_dict, handle)
