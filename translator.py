import tensorflow as tf
import numpy as np
import pickle
from keras_preprocessing.sequence import pad_sequences
from main import Transformer, Encoder, Decoder, MultiHeadAttention, FeedForwardNetwork, PositionalEncoding

# Load tokenizers
def load_tokenizers(src_tokenizer_path, trg_tokenizer_path):
    with open(src_tokenizer_path, 'rb') as f:
        src_tokenizer = pickle.load(f)
    with open(trg_tokenizer_path, 'rb') as f:
        trg_tokenizer = pickle.load(f)
    return src_tokenizer, trg_tokenizer

src_tokenizer, trg_tokenizer = load_tokenizers('preprocessed_data_src_tokenizer.pkl', 'preprocessed_data_trg_tokenizer.pkl')

# Ensure <start> and <end> tokens exist
if '<start>' not in trg_tokenizer.word_index:
    trg_tokenizer.word_index['<start>'] = len(trg_tokenizer.word_index) + 1
if '<end>' not in trg_tokenizer.word_index:
    trg_tokenizer.word_index['<end>'] = len(trg_tokenizer.word_index) + 1

print("<start> token ID:", trg_tokenizer.word_index['<start>'])
print("<end> token ID:", trg_tokenizer.word_index['<end>'])

def get_custom_objects():
    return {
        'Transformer': Transformer,
        'Encoder': Encoder,
        'Decoder': Decoder,
        'MultiHeadAttention': MultiHeadAttention,
        'FeedForwardNetwork': FeedForwardNetwork,
        'PositionalEncoding': PositionalEncoding,
    }

# Load trained Transformer model
checkpoint_filepath = 'transformer_best_model.keras'
transformer = tf.keras.models.load_model(checkpoint_filepath, custom_objects=get_custom_objects())

# Top-k Sampling Function
def top_k_sampling(predictions, k=10):
    values, indices = tf.nn.top_k(predictions, k=k)  # Get top-k predictions
    values = tf.nn.softmax(values).numpy()[0]  # Convert to probabilities
    sampled_index = np.random.choice(indices.numpy()[0], p=values)  # Sample from probabilities
    return sampled_index

# Preprocess input sentence
def preprocess_input(sentence, tokenizer, seq_length=20):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=seq_length, padding='post')
    return tf.convert_to_tensor(padded_sequence, dtype=tf.int32)

# Convert token IDs back to text
def detokenize_output(sequence, tokenizer):
    tokens = tokenizer.sequences_to_texts([sequence])
    return tokens[0]

# Translation with Top-k Sampling
def translate_sentence(input_sentence, transformer, src_tokenizer, trg_tokenizer, seq_length=20):
    # Preprocess input sentence
    encoder_input = preprocess_input(input_sentence, src_tokenizer, seq_length)
    print("Encoder Input Shape:", encoder_input.shape)  # Debugging shape

    # Initialize target sentence with <start> token
    target_sentence = [trg_tokenizer.word_index['<start>']]
    target_tensor = tf.convert_to_tensor([target_sentence], dtype=tf.int32)

    for i in range(seq_length):
        # Create look-ahead mask
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((1, len(target_sentence), len(target_sentence))), -1, 0)

        # Generate predictions
        predictions = transformer(
            src=encoder_input,
            trg=target_tensor,
            src_mask=None,
            trg_mask=look_ahead_mask,
            training=False
        )

        # Use Top-k Sampling to choose the next token
        pred_probs = predictions[:, -1, :]
        next_token = top_k_sampling(pred_probs, k=30)

        # Append the token to the sentence
        target_sentence.append(next_token)

        # Stop decoding if <end> token is generated
        if next_token == trg_tokenizer.word_index['<end>']:
            break

        # Update target tensor
        target_tensor = tf.convert_to_tensor([target_sentence], dtype=tf.int32)

    # Detokenize and return the output sentence
    return detokenize_output(target_sentence[1:], trg_tokenizer)

# Example usage
if __name__ == "__main__":
    input_sentence = "How are you today?"
    print("Translated Sentence:", translate_sentence(input_sentence, transformer, src_tokenizer, trg_tokenizer))
