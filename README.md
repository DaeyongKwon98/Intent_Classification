# Predicting User Intents and Musical Attributes from Music Discovery Conversations

Here is the implementation code and data for the paper titled **"Predicting User Intents and Musical Attributes from Music Discovery Conversations"** by Daeyong Kwon, SeungHeon Doh, Juhan Nam, 2024

<p align="center">
  <img src="https://github.com/user-attachments/assets/a8bfb1dc-856b-4f85-82dd-510cddcc2aeb" alt="Image Load Failed" width="500"/>
  <br>
  <b>Figure 1: Examples of user intents and musical attributes classifcation</b>
</p>

## Setting

The packages and version information required for the implementation are stored in the **requirements.txt** file.

## Implementation

Sparse representation, Word Embedding (Word2Vec), DistilBERT_Probing, DistilBERT_Finetune, and Llama are each implemented in separate .py files. The **functions.py** file should be placed in the same directory to import functions.

Each .py file can be executed by running ```python filename.py```, and the resulting .csv files will be saved in the **"./results"** directory.

For the concatenated setting, you can use ```concat_history``` function in **functions.py**.

## Open Source Material
- [Models](https://huggingface.co/Daeyongkwon98/Music_Conversation_Intent_Classifier/tree/main/models)
- [Dataset](https://huggingface.co/datasets/seungheondoh/cpcd-intent)
- [Huggingface Demo](https://huggingface.co/spaces/Daeyongkwon98/User_Intents_and_Musical_Attributes_Classifier)
