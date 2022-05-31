import pandas as pd 
from tqdm.auto import tqdm
import random 

def transformDataset(training_raw, test=None):

    """
    Transforms multilabel dataset to binary format. 
    If a data point has at least one form of toxicity 
    annotated, the data is considered toxic.
    """
    
    toxic = 0
    non_toxic = 0
    toxicity = []
    data = {'comment' : [],
            'toxic' : [],
            'toxicity' : []
           }

    for i in tqdm(range(len(training_raw))):

        comment = training_raw.loc[i].comment_text
        values = training_raw.loc[i][2:].values
        data['comment'].append(comment)
        
        if max(values) == 1:
            toxic +=1
            data['toxic'].append(1)

        else:
            non_toxic += 1
            data['toxic'].append(0)

        data['toxicity'].append(sum(values))

    return data

def balance_dataset(df, MAX_TOXIC=None, MAX_DF=None):
    
    if not MAX_TOXIC:
        MAX_TOXIC = len(df)
    
    if not MAX_DF:
        MAX_DF = len(df)
        
    data = {'toxic'     : [],
            'non_toxic' : []}

    max_num_tox = 0

    for idx in tqdm(range(len(df))):
        
        row = df.loc[idx]

        if row.toxic == 0:
            
            if len(data['non_toxic']) >= MAX_DF:
                continue
                
            data['non_toxic'].append((row.comment,0))
            
        elif row.toxic == 1:
            
            if len(data['toxic']) >= MAX_TOXIC:
                continue
                
            data['toxic'].append((row.comment,1))
         
    dataset = list(data['toxic'] + data['non_toxic'])
    random.shuffle(dataset)
    
    train_x = [i[0] for i in dataset]
    train_y = [i[1] for i in dataset]
    
    print(f"Num Toxic: {len([i for i in train_y if i ==1])}\nNum Non-toxic: {len([i for i in train_y if i ==0])}")    

    return train_x, train_y
        
def data_processing(path, max_num=None, test=None):

    training_raw = pd.read_csv(path)

    if not max_num:
        max_num = len(training_raw)

    transformed_raw = transformDataset(training_raw[:max_num])
    transformed_df = pd.DataFrame(transformed_raw)
    training_x, training_y = balanceDataset(transformed_df)
    return training_x, training_y
    
