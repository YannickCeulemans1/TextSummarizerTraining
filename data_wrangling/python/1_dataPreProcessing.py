import numpy as np
import pandas as pd
from datetime import datetime as dt
import spacy 
from sentence_transformers import SentenceTransformer
import pickle

### Helper Functions

def text_to_sent_list(text, 
    nlp = spacy.load("en_core_web_lg"), 
    embedder = SentenceTransformer('distilbert-base-nli-mean-tokens'),
    min_len=2):
  
    ''' Returns cleaned article sentences and BERT sentence embeddings'''
    
    # convert to list of sentences
    text = nlp(text)
    sents = list(text.sents)
    # remove short sentences by threshhold                                                                                                
    sents_clean = [sentence.text for sentence in sents if len(sentence)> min_len]
    # remove entries with empty list
    sents_clean = [sentence for sentence in sents_clean if len(sentence)!=0]
    # embed sentences (deafult uses BERT SentenceTransformer)
    sents_embedding= np.array(embedder.encode(sents_clean, convert_to_tensor=True))
    
    return sents_clean, sents_embedding

### Script
output_file = 'data/df_processed.pickle'

df_data = pd.read_csv('dataset/BBCnews.csv')

df = pd.DataFrame(columns=['text_clean','text_embedding','sum_clean','sum_embedding'])
t1 = dt.now()

# clean sentences en embeddings maken en in dataframe steken voor elke row
for index, row in df_data.iterrows():
    print(index)
    text_clean, text_embedding =text_to_sent_list(row['Articles'])
    sum_clean, sum_embedding = text_to_sent_list(row['Summaries'])

    new_row = {'text_clean':text_clean, 'text_embedding':text_embedding,'sum_clean':sum_clean, 'sum_embedding':sum_embedding}
    df = df.append(new_row, ignore_index=True)

# save to pickle
with open(output_file, 'wb') as handle:                                     
    pickle.dump(df, handle)

print(df)
t2=dt.now()
print(t2-t1)


