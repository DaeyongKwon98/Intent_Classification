import warnings
warnings.simplefilter('ignore')
import pandas as pd
from sklearn.metrics import classification_report
import logging
logging.basicConfig(level=logging.ERROR)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from functions import preprocess_data

user_intents = ['initial_query', 'greeting', 'add_filter', 'remove_filter', 'continue', 'accept_response', 'reject_response']
musical_attributes = ['track', 'artist', 'year', 'popularity', 'culture', 'similar_track', 'similar_artist', 'user', 'theme', 'mood', 'genre', 'instrument', 'vocal', 'tempo']
intents_dict = {'user': user_intents, 'music': musical_attributes}

data_dict = preprocess_data("cpcd_intent.csv")

user_train_df = data_dict['user']['train']['dataframe']
user_val_df = data_dict['user']['val']['dataframe']
user_test_df = data_dict['user']['test']['dataframe']

music_train_df = data_dict['music']['train']['dataframe']
music_val_df = data_dict['music']['val']['dataframe']
music_test_df = data_dict['music']['test']['dataframe']

def classification_report_to_df(y_true, y_pred, target_names):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    return report_df

def sparse_representation(train_df, test_df, intents):
	# Seperate text data and labels
	X_train = train_df['content']
	y_train = pd.DataFrame(train_df['intent'].tolist())

	X_test = test_df['content']
	y_test = pd.DataFrame(test_df['intent'].tolist())

	# 1. TF-IDF Vectorize
	tfidf_vectorizer = TfidfVectorizer()
	X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
	X_test_tfidf = tfidf_vectorizer.transform(X_test)

	# 2. Bag-of-Words Vectorize
	bow_vectorizer = CountVectorizer()
	X_train_bow = bow_vectorizer.fit_transform(X_train)
	X_test_bow = bow_vectorizer.transform(X_test)

	# Logistic Regression Model (Use MultiOutputClassifier)
	logistic_model = MultiOutputClassifier(LogisticRegression(max_iter=1000))

	# TF-IDF train and evaluation
	logistic_model.fit(X_train_tfidf, y_train)
	y_pred_tfidf = logistic_model.predict(X_test_tfidf)
	tfidf_result = classification_report_to_df(y_test, y_pred_tfidf, target_names=intents)

	# Bag-of-Words train and evaluation
	logistic_model.fit(X_train_bow, y_train)
	y_pred_bow = logistic_model.predict(X_test_bow)
	bow_result = classification_report_to_df(y_test, y_pred_bow, target_names=intents)

	# Drop unnecessary columns and reset index
	tfidf_result = tfidf_result.drop(columns=['precision', 'recall'])
	tfidf_result.rename(columns={'f1-score': 'f1_tfidf', 'support': 'count'}, inplace=True)
	tfidf_result.reset_index(inplace=True)
	tfidf_result.rename(columns={'index': 'tag'}, inplace=True)

	bow_result = bow_result.drop(columns=['precision', 'recall'])
	bow_result.rename(columns={'f1-score': 'f1_bow', 'support': 'count'}, inplace=True)
	bow_result.reset_index(inplace=True)  # Ensure index alignment with tfidf_result

	# Concatenate 'tag', 'f1_tfidf', and 'f1_bow' columns
	final_result = pd.concat([tfidf_result[['tag', 'f1_tfidf']], bow_result['f1_bow']], axis=1)

	# Round f1 scores to 2 decimal places
	final_result['f1_tfidf'] = final_result['f1_tfidf'].round(2)
	final_result['f1_bow'] = final_result['f1_bow'].round(2)

	# Drop unwanted rows like 'micro avg', 'weighted avg', and 'samples avg'
	final_result = final_result[~final_result['tag'].isin(['micro avg', 'weighted avg', 'samples avg'])]

	return final_result
 
user_sparse_df = sparse_representation(user_train_df, user_test_df, user_intents)
music_sparse_df = sparse_representation(music_train_df, music_test_df, musical_attributes)

user_sparse_df.to_csv("./results/user_sparse.csv", index=False)
music_sparse_df.to_csv("./results/music_sparse.csv", index=False)