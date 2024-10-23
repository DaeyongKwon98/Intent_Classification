import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.ERROR)
from ast import literal_eval
from torch import cuda
import json
from torch.utils.data import TensorDataset

device = 'cuda:0' if cuda.is_available() else 'cpu'

from functions import preprocess_data

class MultiLabelDataset(Dataset):
	def __init__(self, dataframe, tokenizer, max_len):
		self.tokenizer = tokenizer
		self.data = dataframe
		self.text = dataframe.content
		self.targets = self.data.intent
		self.max_len = max_len

	def __len__(self):
		return len(self.text)

	def __getitem__(self, index):
		text = str(self.text[index])
		text = " ".join(text.split())

		tokens = self.tokenizer.tokenize(text)
		if len(tokens) > self.max_len:
			tokens = tokens[-self.max_len:]
		truncated_text = self.tokenizer.convert_tokens_to_string(tokens)

		inputs = self.tokenizer.encode_plus(
			truncated_text,
			None,
			add_special_tokens=True,
			max_length=self.max_len,
			pad_to_max_length=True,
			return_token_type_ids=False,
   			truncation=True
		)
		ids = inputs['input_ids']
		mask = inputs['attention_mask']

		return {
			'ids': torch.tensor(ids, dtype=torch.int),
			'mask': torch.tensor(mask, dtype=torch.int),
			'targets': torch.tensor(self.targets[index], dtype=torch.int)
		}

class DistilBERTClass_Probing(torch.nn.Module):
    def __init__(self, num_intents):
        super(DistilBERTClass_Probing, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1.last_hidden_state
        pooler = hidden_state[:, 0]
        return pooler

class MLP_768(nn.Module):
	def __init__(self, num_intents):
		super(MLP_768, self).__init__()
		self.fc1 = nn.Sequential(
			nn.Linear(768, 64),
			nn.BatchNorm1d(64),
			nn.ReLU(),
		)
		self.fc2 = nn.Linear(64, num_intents)

	def forward(self, x):
		x = self.fc1(x)
		x = self.fc2(x)
		return x

user_intents = ['initial_query', 'greeting', 'add_filter', 'remove_filter', 'continue', 'accept_response', 'reject_response']
musical_attributes = ['track', 'artist', 'year', 'popularity', 'culture', 'similar_track', 'similar_artist', 'user', 'theme', 'mood', 'genre', 'instrument', 'vocal', 'tempo']
intents_dict = {'user': user_intents, 'music': musical_attributes}

device = 'cuda:0' if cuda.is_available() else 'cpu'

data_dict = preprocess_data("cpcd_intent.csv")

MAX_LEN = 128
BATCH_SIZE = 64

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
params = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 0}
num_intents_dict = {'user': 7, 'music': 14}
loss_criterion = torch.nn.BCEWithLogitsLoss()

for data_type in ['user', 'music']:
	training_set = MultiLabelDataset(data_dict[data_type]['train']['dataframe'], tokenizer, MAX_LEN)
	valid_set = MultiLabelDataset(data_dict[data_type]['val']['dataframe'], tokenizer, MAX_LEN)
	test_set = MultiLabelDataset(data_dict[data_type]['test']['dataframe'], tokenizer, MAX_LEN)	
 
	training_loader = DataLoader(training_set, **params)
	valid_loader = DataLoader(valid_set, **params)
	test_loader = DataLoader(test_set, **params)
 
	num_intents = num_intents_dict[data_type]
 
	model = DistilBERTClass_Probing(num_intents)
	model.to(device)
	model.eval()

	train_vectors_list = []
	val_vectors_list = []
	test_vectors_list = []
 
	with torch.no_grad():
		for i, data in tqdm(enumerate(training_loader, 0), leave=False):
			ids = data['ids'].to(device, dtype = torch.int)
			mask = data['mask'].to(device, dtype = torch.int)
			outputs = model(ids, mask)
			train_vectors_list.append(outputs)
		for i, data in tqdm(enumerate(valid_loader, 0), leave=False):
			ids = data['ids'].to(device, dtype = torch.int)
			mask = data['mask'].to(device, dtype = torch.int)
			outputs = model(ids, mask)
			val_vectors_list.append(outputs)
		for i, data in tqdm(enumerate(test_loader, 0), leave=False):
			ids = data['ids'].to(device, dtype = torch.int)
			mask = data['mask'].to(device, dtype = torch.int)
			outputs = model(ids, mask)
			test_vectors_list.append(outputs)

	train_vector = torch.cat(train_vectors_list, dim=0)
	val_vector = torch.cat(val_vectors_list, dim=0)
	test_vector = torch.cat(test_vectors_list, dim=0)
 
	torch.save(train_vector, f'./data/{data_type}_probing_train.pt')
	torch.save(val_vector, f'./data/{data_type}_probing_val.pt')
	torch.save(test_vector, f'./data/{data_type}_probing_test.pt')

	LEARNING_RATE = 2e-4
	NUM_EPOCH = 15
	patience = 2
	count = 0

	training_set = TensorDataset(train_vector, data_dict[data_type]['train']['label'])
	valid_set = TensorDataset(val_vector, data_dict[data_type]['val']['label'])
	test_set = TensorDataset(test_vector, data_dict[data_type]['test']['label'])
 
	training_loader = DataLoader(training_set, **params)
	valid_loader = DataLoader(valid_set, **params)
	test_loader = DataLoader(test_set, **params)

	num_intents = num_intents_dict[data_type]
	model = MLP_768(num_intents)
	model.to(device)

	optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
	scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
	loss_criterion = torch.nn.BCEWithLogitsLoss()
 
	train_loss_list = []
	valid_loss_list = []	

	min_loss = 100.0
	# Train
	print("Training Start")
	for epoch in range(NUM_EPOCH):
		train_loss = 0.0
		model.train()
		for i, (batch_x, batch_y) in tqdm(enumerate(training_loader, 0), leave=False):
			batch_x = batch_x.to(device, dtype=torch.float)
			batch_y = batch_y.to(device, dtype=torch.float)
			outputs = model(batch_x)
			optimizer.zero_grad()
			loss = loss_criterion(outputs, batch_y)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
		scheduler.step()
		train_loss /= i
		
		# Validation
		model.eval()
		probability_outputs=[]
		valid_loss = 0.0
		with torch.no_grad():
			for i, (batch_x, batch_y) in tqdm(enumerate(valid_loader, 0), leave=False):
				batch_x = batch_x.to(device, dtype=torch.float)
				batch_y = batch_y.to(device, dtype=torch.float)
				outputs = model(batch_x)
				valid_loss += loss_criterion(outputs, batch_y).item()
			valid_loss /= i
		print(f"Epoch:{epoch+1}, Train Loss: {round(train_loss,4)}, Valid Loss: {round(valid_loss,4)}, lr: {LEARNING_RATE*(0.9)**epoch}")
		train_loss_list.append(round(train_loss,4))
		valid_loss_list.append(round(valid_loss,4))
 
		# Save model
		if valid_loss < min_loss:
			min_loss = valid_loss
			count = 0
   
			model_name = f"./models/{data_type}_probing_model.pth"
			torch.save(model.state_dict(), model_name)
			print(f"{data_type} model saved with valid loss {round(min_loss,4)}")
		else:
			count += 1
			if count==patience: break
 
	# Load Best Model
	model = MLP_768(num_intents)
	model.load_state_dict(torch.load(f"./models/{data_type}_probing_model.pth"))
	model.to(device)

	# Find best threshold
	model.eval()
	labels_list = []
	probability_outputs=[]
	with torch.no_grad():
		for i, (batch_x, batch_y) in tqdm(enumerate(valid_loader, 0), leave=False):
			batch_x = batch_x.to(device, dtype=torch.float)
			batch_y = batch_y.to(device, dtype=torch.float)
			outputs = model(batch_x)
			labels_list.extend(batch_y.cpu().detach().numpy().tolist())
			probability_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

	labels = np.array(labels_list)
	probability_outputs = np.array(probability_outputs)

	best_thresholds = np.zeros(num_intents)
	thresholds = [(i+1)/100 for i in range(100)]
	for label_idx in range(num_intents):
		best_label_f1 = 0.0
		for threshold in thresholds:
			binary_preds = (probability_outputs[:, label_idx] >= threshold).astype(int)
			f1 = f1_score(labels[:, label_idx], binary_preds)
			if f1 > best_label_f1:
				best_label_f1 = f1
				best_thresholds[label_idx] = threshold

	# Test with best thresholds
	model.eval()
	labels_list = []
	probability_outputs=[]
	for i, (batch_x, batch_y) in tqdm(enumerate(test_loader, 0), leave=False):
		batch_x = batch_x.to(device, dtype=torch.float)
		batch_y = batch_y.to(device, dtype=torch.float)
		outputs = model(batch_x)	
		labels_list.extend(batch_y.cpu().detach().numpy().tolist())
		probability_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

	labels = np.array(labels_list)
	probability_outputs = np.array(probability_outputs)
 
	binary_outputs = (probability_outputs >= best_thresholds).astype(int)
	binary_outputs[np.all(binary_outputs == False, axis=1), -1] = True
 
	report = classification_report(labels[:, :-1], binary_outputs[:, :-1], output_dict=True)
	df_report = pd.DataFrame(report).transpose()
	df_report.reset_index(inplace=True)
	del df_report['precision']
	del df_report['recall']
	del df_report['support']
 
	df_report.rename(columns={'index': 'tag'}, inplace=True)
	df_report.loc[:num_intents-2, 'tag'] = intents_dict[data_type][:-1]
	df_report['f1-score'] = df_report['f1-score'].round(2)
	df_report = df_report[df_report['tag'].apply(lambda x: x not in ['micro avg', 'weighted avg', 'samples avg'])]
	df_report.to_csv(f"./results/{data_type}_probing.csv", index=False)