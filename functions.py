import warnings
warnings.simplefilter('ignore')
import pandas as pd
import torch
import logging
logging.basicConfig(level=logging.ERROR)
from ast import literal_eval
import json
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

user_intents = ['initial_query', 'greeting', 'add_filter', 'remove_filter', 'continue', 'accept_response', 'reject_response']
musical_attributes = ['track', 'artist', 'year', 'popularity', 'culture', 'similar_track', 'similar_artist', 'user', 'theme', 'mood', 'genre', 'instrument', 'vocal', 'tempo']

def preprocess_data(data_path):
    df = pd.read_csv(data_path, encoding='unicode_escape')
    df['intent'] = df['intent'].apply(literal_eval)
    df['music_attribute'] = df['music_attribute'].apply(literal_eval)

    with open('error_dialog_id.json', 'r') as json_file:
        error_dialog_id = json.load(json_file)
    df = df[df['dialog_id'].apply(lambda x: x not in error_dialog_id)]

    # Change user intents less than 20 into others.
    df["intent"] = df["intent"].apply(lambda x: ["others" if item in ["item_attribute_answer", "item_attribute_question"] else item for item in x])

    def remove_others_if_not_alone(intents):
        if 'others' in intents and len(intents) > 1:
            intents.remove('others')
        return intents
    df['intent'] = df['intent'].apply(remove_others_if_not_alone)

    # initial_query can't exist with [remove_filter, continue, accept_response, reject_response, others]
    def preprocess_initial(row):
        if 'initial_query' in row['intent']:
            for intent_to_remove in ['remove_filter', 'continue', 'accept_response', 'reject_response', 'others']:
                if intent_to_remove in row['intent']:
                    row['intent'].remove(intent_to_remove)
        return row
    df = df.apply(preprocess_initial, axis=1)

    def concat_previous_1_rows(group):
        if len(group) < 1:
            return pd.DataFrame()
        group = group.copy()
        group['content'] = group['content'].shift(1).fillna('') + '. ' + group['content']
        group['content'].iloc[0] = group['content'].iloc[0].lstrip('. ')
        return group

    user_df = df.groupby('dialog_id').apply(concat_previous_1_rows).reset_index(drop=True)

    user_df = user_df[user_df['role']=='user']

    del user_df['role']
    del user_df['music_attribute']

    def encode_intents(intent_list, intents):
        return [1 if intent in intent_list else 0 for intent in intents]

    user_df.loc[:, 'intent'] = user_df['intent'].apply(lambda x: encode_intents(x, user_intents))
    user_df = user_df.reset_index(drop=True)

    music_df = df[['index','dialog_id', 'role', 'content', 'music_attribute']]
    music_df.loc[:, 'music_attribute'] = music_df['music_attribute'].apply(lambda x: encode_intents(x, musical_attributes))
    music_df.rename(columns={'music_attribute': 'intent'}, inplace=True)
    music_df = music_df.reset_index(drop=True)

    user_y = torch.stack([torch.tensor(item) for item in user_df['intent']])
    music_y = torch.stack([torch.tensor(item) for item in music_df['intent']])

    # Train, Valid Split
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in msss.split(user_df['content'].values, user_y):
        user_train_df, user_val_df = user_df.iloc[train_index], user_df.iloc[test_index]
        user_train_y, user_val_y = user_y[train_index], user_y[test_index]

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

    for train_index, test_index in msss.split(user_val_df['content'].values, user_val_y):
        user_val_df, user_test_df = user_val_df.iloc[train_index], user_val_df.iloc[test_index]
        user_val_y, user_test_y = user_val_y[train_index], user_val_y[test_index]

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in msss.split(music_df['content'].values, music_y):
        music_train_df, music_val_df = music_df.iloc[train_index], music_df.iloc[test_index]
        music_train_y, music_val_y = music_y[train_index], music_y[test_index]

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

    for train_index, test_index in msss.split(music_val_df['content'].values, music_val_y):
        music_val_df, music_test_df = music_val_df.iloc[train_index], music_val_df.iloc[test_index]
        music_val_y, music_test_y = music_val_y[train_index], music_val_y[test_index]

    user_train_df = user_train_df.reset_index(drop=True)
    user_val_df = user_val_df.reset_index(drop=True)
    user_test_df = user_test_df.reset_index(drop=True)

    music_train_df = music_train_df.reset_index(drop=True)
    music_val_df = music_val_df.reset_index(drop=True)
    music_test_df = music_test_df.reset_index(drop=True)

    data_dict = {
        'user': {
            'train': {
                'dataframe': user_train_df,
                'label': user_train_y
            },
            'val': {
                'dataframe': user_val_df,
                'label': user_val_y
            },
            'test': {
                'dataframe': user_test_df,
                'label': user_test_y
            }
        },
        'music': {
            'train': {
                'dataframe': music_train_df,
                'label': music_train_y
            },
            'val': {
                'dataframe': music_val_df,
                'label': music_val_y
            },
            'test': {
                'dataframe': music_test_df,
                'label': music_test_y
            }
        }
    }
    
    return data_dict