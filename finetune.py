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
from sklearn.metrics import classification_report
import logging
logging.basicConfig(level=logging.ERROR)
from torch import cuda

from functions import preprocess_data

device = 'cuda:0' if cuda.is_available() else 'cpu'

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

class DistilBERTClass(nn.Module):
	def __init__(self, num_intents):
		super(DistilBERTClass, self).__init__()
		self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
		self.fc1 = nn.Sequential(
			nn.Linear(768, 64),
			nn.BatchNorm1d(64),
			nn.ReLU(),
		)
		self.fc2 = nn.Sequential(
			nn.Linear(64, num_intents)
		)

	def forward(self, input_ids, attention_mask):
		output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
		hidden_state = output_1[0]
		pooler = hidden_state[:, 0]
		pooler = self.fc1(pooler)
		output = self.fc2(pooler)
		return output

MAX_LEN = 128
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
NUM_EPOCH = 15
patience = 2
count = 0

user_intents = ['initial_query', 'greeting', 'add_filter', 'remove_filter', 'continue', 'accept_response', 'reject_response']
musical_attributes = ['track', 'artist', 'year', 'popularity', 'culture', 'similar_track', 'similar_artist', 'user', 'theme', 'mood', 'genre', 'instrument', 'vocal', 'tempo']

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
params = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 4}
intents_dict = {"user": user_intents, "music": musical_attributes}
num_intents_dict = {'user': 7, 'music': 14}

data_dict = preprocess_data("cpcd_intent.csv")

for data_type in ['user', 'music']:
	training_set = MultiLabelDataset(data_dict[data_type]['train']['dataframe'], tokenizer, MAX_LEN)
	valid_set = MultiLabelDataset(data_dict[data_type]['val']['dataframe'], tokenizer, MAX_LEN)
	test_set = MultiLabelDataset(data_dict[data_type]['test']['dataframe'], tokenizer, MAX_LEN)	
 
	training_loader = DataLoader(training_set, **params)
	valid_loader = DataLoader(valid_set, **params)
	test_loader = DataLoader(test_set, **params)

	num_intents = num_intents_dict[data_type]
	model = DistilBERTClass(num_intents)
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
		for i,data in tqdm(enumerate(training_loader, 0), leave=False):
			ids = data['ids'].to(device, dtype = torch.int)
			mask = data['mask'].to(device, dtype = torch.int)
			targets = data['targets'].to(device, dtype = torch.float)
			outputs = model(ids, mask)
			optimizer.zero_grad()
			loss = loss_criterion(outputs, targets)
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
			for i, data in tqdm(enumerate(valid_loader, 0), leave=False):
				ids = data['ids'].to(device, dtype = torch.int)
				mask = data['mask'].to(device, dtype = torch.int)
				targets = data['targets'].to(device, dtype = torch.float)
				outputs = model(ids, mask)
				valid_loss += loss_criterion(outputs, targets).item()
			valid_loss /= i
		print(f"Epoch:{epoch+1}, Train Loss: {round(train_loss,4)}, Valid Loss: {round(valid_loss,4)}, lr: {LEARNING_RATE*(0.9)**epoch}")
		train_loss_list.append(round(train_loss,4))
		valid_loss_list.append(round(valid_loss,4))
 
		# Save model
		if valid_loss < min_loss:
			min_loss = valid_loss
			count = 0
   
			model_name = f"./models/{data_type}_finetune_model.pth"
			torch.save(model.state_dict(), model_name)
			print(f"{data_type} model saved with valid loss {round(min_loss,4)}")
		else:
			count += 1
			if count==patience: break
 
	# Load Best Model
	model = DistilBERTClass(num_intents)
	model.load_state_dict(torch.load(f"./models/{data_type}_finetune_model.pth"))
	model.to(device)

	# Test with best model
	print("Test Start")
	test_loss = 0.0
	model.eval()
	probability_outputs=[]
	with torch.no_grad():
		for i, data in tqdm(enumerate(test_loader, 0), leave=False):
			ids = data['ids'].to(device, dtype = torch.int)
			mask = data['mask'].to(device, dtype = torch.int)
			targets = data['targets'].to(device, dtype = torch.float)
			outputs = model(ids, mask)
			test_loss += loss_criterion(outputs, targets).item()
			probability_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
		test_loss /= i
		print(f"Test loss: {round(test_loss,4)}")
	probability_outputs = np.array(probability_outputs)
    
	binary_outputs = (probability_outputs >= 0.5)
	binary_outputs[np.all(binary_outputs == False, axis=1), -1] = True # Label as 'others(none)' if everything is 0

	# Calculate Result
	report = classification_report(data_dict[data_type]['test']['label'], binary_outputs, output_dict=True)
	df_report = pd.DataFrame(report).transpose()
	df_report.reset_index(inplace=True)
	del df_report['precision']
	del df_report['recall']
	del df_report['support']

	df_report.rename(columns={'index': 'tag'}, inplace=True)
	df_report.loc[:num_intents-1, 'tag'] = intents_dict[data_type]
	df_report['f1-score'] = df_report['f1-score'].round(2)
	df_report = df_report[df_report['tag'].apply(lambda x: x not in ['micro avg', 'weighted avg', 'samples avg'])]
	df_report.to_csv(f"./results/{data_type}_finetune.csv", index=False)