import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging
logging.basicConfig(level=logging.ERROR)
from torch import cuda
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gensim.downloader as api

from functions import preprocess_data

device = 'cuda:1' if cuda.is_available() else 'cpu'

user_intents = ['initial_query', 'greeting', 'add_filter', 'remove_filter', 'continue', 'accept_response', 'reject_response']
music_attributes = ['track', 'artist', 'year', 'popularity', 'culture', 'similar_track', 'similar_artist', 'user', 'theme', 'mood', 'genre', 'instrument', 'vocal', 'tempo']
intents_dict = {'user': user_intents, 'music': music_attributes}

data_dict = preprocess_data("cpcd_intent.csv")

user_train_df = data_dict['user']['train']['dataframe']
user_val_df = data_dict['user']['val']['dataframe']
user_test_df = data_dict['user']['test']['dataframe']

music_train_df = data_dict['music']['train']['dataframe']
music_val_df = data_dict['music']['val']['dataframe']
music_test_df = data_dict['music']['test']['dataframe']

model = api.load('word2vec-google-news-300')

class WhitespaceTokenizer:
	def __init__(self, do_lower_case=True):
		self.do_lower_case = do_lower_case

	def tokenize(self, text):
		if self.do_lower_case:
			text = text.lower()
		return text.split()
	
tokenizer = WhitespaceTokenizer(do_lower_case=True)

def get_word_vector(token):
	try:
		return model[token]
	except KeyError:
		return None

def sentence_to_vector(text):
	tokens = tokenizer.tokenize(text)
	vectors = [get_word_vector(token) for token in tokens if get_word_vector(token) is not None]

	if vectors:
		sentence_vector = np.mean(vectors, axis=0)
	else:
		sentence_vector = np.zeros(300)
	return sentence_vector

user_train_vector = np.array(user_train_df['content'].apply(lambda x: sentence_to_vector(x)).values.tolist())
user_val_vector = np.array(user_val_df['content'].apply(lambda x: sentence_to_vector(x)).values.tolist())
user_test_vector = np.array(user_test_df['content'].apply(lambda x: sentence_to_vector(x)).values.tolist())

music_train_vector = np.array(music_train_df['content'].apply(lambda x: sentence_to_vector(x)).values.tolist())
music_val_vector = np.array(music_val_df['content'].apply(lambda x: sentence_to_vector(x)).values.tolist())
music_test_vector = np.array(music_test_df['content'].apply(lambda x: sentence_to_vector(x)).values.tolist())

class MLP(nn.Module):
	def __init__(self, num_intents):
		super(MLP, self).__init__()
		self.fc1 = nn.Sequential(
			nn.Linear(300, 64),
			nn.BatchNorm1d(64),
			nn.ReLU()
		)
		self.fc2 = nn.Sequential(
			nn.Linear(64, num_intents)
		)
	
	def forward(self, x):
		out = self.fc1(x)
		out = self.fc2(out)
		return out

# Parameters
num_epochs = 15
batch_size = 64
learning_rate = 2e-4
patience = 2

vector_dict = {
	'user': {'train': user_train_vector, 'val': user_val_vector, 'test': user_test_vector},
	'music': {'train': music_train_vector, 'val': music_val_vector, 'test': music_test_vector}
}
num_intents_dict = {'user': 7, 'music': 14}

for data_type in ['user', 'music']:
	X_train = vector_dict[data_type]['train']
	X_val = vector_dict[data_type]['val']
	X_test = vector_dict[data_type]['test']
	y_train = data_dict[data_type]['train']['label'].numpy()
	y_val = data_dict[data_type]['val']['label'].numpy()
	y_test = data_dict[data_type]['test']['label'].numpy()
	
	num_intents = num_intents_dict[data_type]
	
	model = MLP(num_intents)
	loss_criterion = torch.nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
	
	X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
	y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
	X_valid_tensor = torch.tensor(X_val, dtype=torch.float32)
	y_valid_tensor = torch.tensor(y_val, dtype=torch.float32)
	X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
	y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

	train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
	valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
	test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	min_loss = 1000.0
	
	for epoch in range(num_epochs):
		train_loss = 0.0
		model.train()
		for i, (batch_x, batch_y) in tqdm(enumerate(train_loader,0), leave=False):
			optimizer.zero_grad()
			outputs = model(batch_x)
			loss = loss_criterion(outputs, batch_y)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
		scheduler.step()
		train_loss /= i
	
		# Show Validation Loss
		model.eval()
		valid_loss = 0.0
		with torch.no_grad():
			all_outputs = []
			all_labels = []
			for i, (batch_x, batch_y) in tqdm(enumerate(valid_loader, 0), leave=False):
				outputs = model(batch_x)
				all_outputs.append(outputs)
				all_labels.append(batch_y)
				valid_loss += loss_criterion(outputs, batch_y).item()
			valid_loss /= i
   
		all_outputs = torch.sigmoid(torch.cat(all_outputs, dim=0))
		all_labels = torch.cat(all_labels, dim=0)
		print(f"Epoch:{epoch+1}, Train Loss: {round(train_loss,4)}, Valid Loss: {round(valid_loss,4)}")
		predicted = (all_outputs >= 0.5).float()
		accuracy = accuracy_score(all_labels.numpy(), predicted.numpy())
		print(f'Validation Accuracy: {accuracy:.4f}')
  
		if valid_loss < min_loss:
			min_loss = valid_loss
			count = 0
		else:
			count += 1
			if count==patience: break
	
	# Find best threshold with validation set
	model.eval()
	labels = []
	probability_outputs=[]
	with torch.no_grad():
		all_outputs = []
		all_labels = []
		for i, (batch_x, batch_y) in tqdm(enumerate(valid_loader, 0), leave=False):
			outputs = model(batch_x)
			all_outputs.append(outputs)
			all_labels.append(batch_y)

	all_outputs = torch.sigmoid(torch.cat(all_outputs, dim=0))
	all_labels = torch.cat(all_labels, dim=0)
	best_thresholds = np.zeros(num_intents)
	thresholds = [(i+1)/100 for i in range(100)]
	for label_idx in range(num_intents):
		best_label_f1 = 0.0
		for threshold in thresholds:
			binary_preds = (all_outputs[:, label_idx] >= threshold).numpy().astype(int)
			f1 = f1_score(all_labels[:, label_idx], binary_preds)
			if f1 > best_label_f1:
				best_label_f1 = f1
				best_thresholds[label_idx] = threshold
 
  	# Test with best model
	probability_outputs=[]
	with torch.no_grad():
		all_outputs = []
		all_labels = []
		for i, (batch_x, batch_y) in tqdm(enumerate(test_loader, 0), leave=False):
			outputs = model(batch_x)
			all_outputs.append(outputs)
			all_labels.append(batch_y)
   
	all_outputs = torch.sigmoid(torch.cat(all_outputs, dim=0))
	all_labels = torch.cat(all_labels, dim=0)
	binary_outputs = (all_outputs.numpy() >= best_thresholds).astype(int)
	binary_outputs[np.all(binary_outputs == False, axis=1), -1] = True
 
	# Calculate Result
	report = classification_report(all_labels, binary_outputs, output_dict=True)
	df_report = pd.DataFrame(report).transpose()
	df_report.reset_index(inplace=True)
	del df_report['precision']
	del df_report['recall']
	del df_report['support']
	df_report.rename(columns={'index': 'tag'}, inplace=True)
	df_report.loc[:num_intents-1, 'tag'] = intents_dict[data_type]
	df_report['f1-score'] = df_report['f1-score'].round(2)
	df_report = df_report[~df_report['tag'].isin(['micro avg', 'weighted avg', 'samples avg'])]
	df_report.to_csv(f"./results/{data_type}_word2vec.csv", index=False)