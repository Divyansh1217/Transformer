import pickle
from keras_preprocessing.text import Tokenizer

# Load the tokenizer
with open('Folder/preprocessed_data_trg_tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Check the type of the tokenizer
print(type(tokenizer))  # Should print <class 'keras.preprocessing.text.Tokenizer'>

# Access tokenizer's attributes directly
print("Word Index (first 10):", dict(list(tokenizer.word_index.items())[:10]))  # First 10 words and their indices
print("Number of Words:", tokenizer.num_words)  # Total words to consider
print("Document Count:", tokenizer.document_count)  # Number of documents processed
