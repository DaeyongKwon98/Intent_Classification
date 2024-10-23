import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
import logging
logging.basicConfig(level=logging.ERROR)
from torch import cuda
import json
import re
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from transformers import pipeline

from functions import preprocess_data

def load_llama_models(model_size):
    if model_size=="1": model_id = "meta-llama/Llama-3.2-1B-Instruct"
    elif model_size=="3": model_id = "meta-llama/Llama-3.2-3B-Instruct"
    elif model_size=="8": model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        pad_token_id=50256,
    )
    print("Model Load Finish")
    return pipe

def load_prompts(prompt_path, prompt_type):
    with open(prompt_path, 'r') as json_file:
	    return json.load(json_file)[prompt_type]

def calculate(data_dict, prompt, model, data_type, csv_path):
    df = data_dict[data_type]['test']['dataframe']
    y = data_dict[data_type]['test']['label']
    
    if data_type=="user": intents = user_intents
    else: intents = musical_attributes

    predicted_vectors = []

    input_texts = df['content'].values
    for i in range(len(input_texts)):
        current_prompt = prompt + f"""
        Input: "{input_texts[i]}"
        Output: [
        """

        output = model(current_prompt, max_new_tokens=200, temperature=0.1)

        generated_text = output[0]['generated_text'].strip()

        match = re.search(rf'Input: "{re.escape(input_texts[i])}"\s*Output:\s*\[(.*?)\]', generated_text, re.DOTALL)

        if match:
            relevant_attributes = match.group(1).strip()
            relevant_attributes_list = [attr.strip().strip('"') for attr in relevant_attributes.split(",")]
        else:
            relevant_attributes_list = []

        relevant_attributes_list = [re.sub(r'^[^a-zA-Z0-9_]+|[^a-zA-Z0-9_]+$', '', s) for s in relevant_attributes_list]

        binary_vector = [1 if attr in relevant_attributes_list else 0 for attr in intents]

        predicted_vectors.append(binary_vector)
        
        print(f"{i+1} finish")

    predicted_vectors = np.array(predicted_vectors)
    ground_truth = np.array(y)

    f1_scores_per_label = f1_score(ground_truth, predicted_vectors, average=None)
    f1_score_overall = f1_score(ground_truth, predicted_vectors, average='macro')

    f1_df = pd.DataFrame({
        'Labels': intents,
        'F1 Score': f1_scores_per_label
    })

    f1_df.loc[len(f1_df.index)] = ['Overall (Macro-Averaged)', f1_score_overall]
    f1_df.to_csv(csv_path, index=False)
    print(f"{csv_path} is generated!")

def compare_predictions(data_dict, prompt, model, data_type):
    results = []
    
    df = data_dict[data_type]['test']['dataframe']
    y = data_dict[data_type]['test']['label']
    
    if data_type=="user": intents = user_intents
    else: intents = musical_attributes
    
    input_texts = df['content'].values
    for i in range(len(input_texts)):

        current_prompt = prompt + f"""
        Input: "{input_texts[i]}"
        Output: [
        """

        output = model(current_prompt, max_new_tokens=200, temperature=0.1)

        generated_text = output[0]['generated_text'].strip()

        match = re.search(rf'Input: "{re.escape(input_texts[i])}"\s*Output:\s*\[(.*?)\]', generated_text, re.DOTALL)

        if match:
            relevant_attributes = match.group(1).strip()
            relevant_attributes_list = [attr.strip().strip('"') for attr in relevant_attributes.split(",")]
        else:
            relevant_attributes_list = []

        binary_vector = [1 if attr in relevant_attributes_list else 0 for attr in intents]

        ground_truth = [intents[k] for k in range(len(binary_vector)-1) if y[i][k] == 1]
        prediction = [intents[k] for k in range(len(binary_vector)-1) if binary_vector[k] == 1]

        results.append({
            "Input": input_texts[i],
            "Ground Truth": ground_truth,
            "Prediction": prediction
        })

    return pd.DataFrame(results)

################
user_intents = ['initial_query', 'greeting', 'add_filter', 'remove_filter', 'continue', 'accept_response', 'reject_response']
musical_attributes = ['track', 'artist', 'year', 'popularity', 'culture', 'similar_track', 'similar_artist', 'user', 'theme', 'mood', 'genre', 'instrument', 'vocal', 'tempo']

# Settings
device = 'cuda:0' if cuda.is_available() else 'cpu'

model_size = '1' # '1' or '3' or '8'
data_type = 'user' # 'user' or 'music'
shot_type = '0' # '0' or '5'

csv_path = f"./results/{model_size}B_{data_type}_{shot_type}.csv"
data_dict = preprocess_data("cpcd_intent.csv")

model = load_llama_models(model_size)
prompt = load_prompts("prompts.json", f"{data_type}_{shot_type}")

# Generate .csv file
calculate(data_dict, prompt, model, data_type, csv_path)

# Compare predictions with ground truth
difference = compare_predictions(data_dict, prompt, model, data_type)
print(difference)