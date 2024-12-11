import numpy as np
import tensorflow as tf
from keras import layers, Sequential
from keras_preprocessing.sequence import pad_sequences

# Scaled Dot-Product Attention
def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    d_k = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output, attention_weights

# Multi-Head Attention Layer
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.query_dense = layers.Dense(d_model)
        self.key_dense = layers.Dense(d_model)
        self.value_dense = layers.Dense(d_model)
        self.final_dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]

        query = self.split_heads(self.query_dense(query), batch_size)
        key = self.split_heads(self.key_dense(key), batch_size)
        value = self.split_heads(self.value_dense(value), batch_size)

        attention, weights = scaled_dot_product_attention(query, key, value, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.d_model))
        output = self.final_dense(concat_attention)
        return output, weights

# Feed Forward Network
class FeedForwardNetwork(layers.Layer):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.ffn = Sequential([
            layers.Dense(d_ff, activation='relu'),
            layers.Dense(d_model)
        ])

    def call(self, x):
        return self.ffn(x)

# Positional Encoding
def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model) // 2)) / d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = positional_encoding(position, d_model)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

# Encoder Layer
class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, src_mask=None, training=False):
        attn_output, _ = self.attention(x, x, x, mask=src_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

# Encoder
class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, max_pos, dropout_rate):
        super(Encoder, self).__init__()
        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_pos, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, src_mask=None, training=True):
        x = self.embedding(x)
        x += self.pos_encoding[:, :tf.shape(x)[1], :]
        x = self.dropout(x, training=training)

        for layer in self.enc_layers:
            x = layer(x, src_mask=src_mask, training=training)

        return x

# Decoder Layer
class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, x, enc_output, trg_mask, src_mask, training):
        self_attn_output, _ = self.self_attention(x, x, x, mask=trg_mask)
        self_attn_output = self.dropout1(self_attn_output, training=training)
        out1 = self.layernorm1(x + self_attn_output)

        enc_dec_attn_output, _ = self.enc_dec_attention(out1, enc_output, enc_output, mask=src_mask)
        enc_dec_attn_output = self.dropout2(enc_dec_attn_output, training=training)
        out2 = self.layernorm2(out1 + enc_dec_attn_output)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3

# Decoder
class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, target_vocab_size, max_pos, dropout_rate):
        super(Decoder, self).__init__()
        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_pos, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, enc_output, src_mask=None, trg_mask=None, training=True):
        x = self.embedding(x)
        x += self.pos_encoding[:, :tf.shape(x)[1], :]
        x = self.dropout(x, training=training)

        for layer in self.dec_layers:
            x = layer(x, enc_output, trg_mask=trg_mask, src_mask=src_mask, training=training)

        return x

# Transformer
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_pos, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, max_pos, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, target_vocab_size, max_pos, dropout_rate)
        self.final_layer = layers.Dense(target_vocab_size)

    def call(self, src, trg, src_mask=None, trg_mask=None, training=False):
        enc_output = self.encoder(src, src_mask=src_mask, training=training)
        dec_output = self.decoder(trg, enc_output, src_mask=src_mask, trg_mask=trg_mask, training=training)
        return self.final_layer(dec_output)



    @classmethod
    def from_config(cls, config):
        config.pop('trainable', None)
        config.pop('dtype', None)
        return cls(
                    num_layers=config.get('num_layers', 4),  # Default value if not found
                    d_model=config.get('d_model', 128),  # Default value if not found
                    num_heads=config.get('num_heads', 8),  # Default value if not found
                    d_ff=config.get('d_ff', 512),  # Default value if not found
                    input_vocab_size=config.get('input_vocab_size', 8000),  # Default value if not found
                    target_vocab_size=config.get('target_vocab_size', 8000),  # Default value if not found
                    max_pos=config.get('max_pos', 1000)  # Default value if not found
        )
def save_preprocessed_data(src_seq, trg_seq, src_tokenizer, trg_tokenizer, file_prefix="preprocessed_data"):
    with open(f"{file_prefix}_src_train.pkl", "wb") as f:
        pickle.dump(src_seq, f)
    with open(f"{file_prefix}_trg_train.pkl", "wb") as f:
        pickle.dump(trg_seq, f)
    with open(f"{file_prefix}_src_tokenizer.pkl", "wb") as f:
        pickle.dump(src_tokenizer, f)
    with open(f"{file_prefix}_trg_tokenizer.pkl", "wb") as f:
        pickle.dump(trg_tokenizer, f)

# Load preprocessed data
def load_preprocessed_data(file_prefix="Folder/preprocessed_data"):
    with open(f"{file_prefix}_src_train.pkl", "rb") as f:
        src_seq = pickle.load(f)
    with open(f"{file_prefix}_trg_train.pkl", "rb") as f:
        trg_seq = pickle.load(f)
    with open(f"{file_prefix}_src_tokenizer.pkl", "rb") as f:
        src_tokenizer = pickle.load(f)
    with open(f"{file_prefix}_trg_tokenizer.pkl", "rb") as f:
        trg_tokenizer = pickle.load(f)
    return src_seq, trg_seq, src_tokenizer, trg_tokenizer
# Main NLP
import pickle
if __name__=="__main__":
    from keras_preprocessing.text import Tokenizer
    from datasets import load_dataset
    try:
        # Load preprocessed data
        src_train, trg_train, src_tokenizer, trg_tokenizer = load_preprocessed_data()
        print("Loaded preprocessed data.")
    except FileNotFoundError:
        # Load and preprocess dataset
        print("Preprocessed data not found. Loading dataset...")
        dataset = load_dataset("wmt16", "ru-en")
        train_dataset = dataset["train"]
        src_train = [loop["translation"]["en"] for loop in train_dataset]
        trg_train = [loop["translation"]["ru"] for loop in train_dataset]
        
        input_vocab_size = 10000
        target_vocab_size = 10000
        # Toke
        src_tokenizer = Tokenizer(num_words=input_vocab_size, oov_token="<OOV>")
        trg_tokenizer = Tokenizer(num_words=target_vocab_size, oov_token="<OOV>")
        src_train, trg_train, src_tokenizer, trg_tokenizer = save_preprocessed_data(src_train, trg_train, src_tokenizer, trg_tokenizer)
    src_train = list(map(str, src_train))  # Convert each item to a string
    trg_train = list(map(str, trg_train))
    src_tokenizer.fit_on_texts(src_train)
    trg_tokenizer.fit_on_texts(trg_train)
    src_seq=src_tokenizer.texts_to_sequences(src_train)
    trg_seq=trg_tokenizer.texts_to_sequences(trg_train)
    seq_length=20
    # Padding
    input_vocab_size=10000
    target_vocab_size=10000
    src_seq = [[token if token < input_vocab_size else 1 for token in seq] for seq in src_seq]
    trg_seq = [[token if token < target_vocab_size else 1 for token in seq] for seq in trg_seq]
    src_padding=pad_sequences(src_seq,maxlen=seq_length,padding='post',truncating='post')
    trg_padding=pad_sequences(trg_seq,maxlen=seq_length,padding='post',truncating='post')

    trg_input = trg_padding[:, :-1]
    trg_target = trg_padding[:, 1:]

    # Encoder input
    x=tf.convert_to_tensor(src_padding,dtype=tf.int32)
    # Decoder input
    Y_input = tf.convert_to_tensor(trg_input, dtype=tf.int32) 
    # Decoder Target
    Y_target = tf.convert_to_tensor(trg_target, dtype=tf.int32)
    # Just for error
    print(f"X Shape: {x.shape}, Y_input Shape: {Y_input.shape}, Y_target Shape: {Y_target.shape}") 
    # Training Steps
    loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')
    print(loss_object)

    # Transformer instance
    num_layers = 4
    d_model = 256
    num_heads = 8
    d_ff = 512
    input_vocab_size = 10000
    target_vocab_size = 10000
   
    max_pos = 5000
    dropout_rate = 0.5

    transformer=Transformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_pos, dropout_rate)
    print(transformer.get_config())

    def loss_function(real,pred):
        mask=tf.math.logical_not(tf.math.equal(real,0))
        loss=loss_object(real,pred)
        mask=tf.cast(mask,dtype=loss.dtype)
        loss*=mask
        return tf.reduce_mean(loss)
    optimizer = tf.keras.optimizers.Adam()

    # for look ahead mask
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask 

    @tf.function
    def train_step(encoder_input, decoder_input, decoder_target):
    # Create masks
        look_ahead_mask = create_look_ahead_mask(tf.shape(decoder_input)[1])
        with tf.GradientTape() as tape:
            predictions = transformer(
                src=encoder_input,
                trg=decoder_input,
                src_mask=None,
                trg_mask=None,
                training=True
            )
            loss = loss_function(decoder_target, predictions)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        return loss
    def get_custom_objects():
        return {
        'Transformer': Transformer,
        'Encoder': Encoder,
        'Decoder': Decoder,
        'MultiHeadAttention': MultiHeadAttention,
        'FeedForwardNetwork': FeedForwardNetwork,
        'PositionalEncoding': PositionalEncoding,
    }
    checkpoint_filepath = 'transformer_best_model.keras'
    import os
    # Reload the saved model
    if os.path.exists(checkpoint_filepath):
        print("Loading saved model...")
        transformer = tf.keras.models.load_model(checkpoint_filepath, custom_objects=get_custom_objects())
    else:
        print("Saved model not found. Creating a new model...")
        transformer = Transformer(
            num_layers, d_model, num_heads, d_ff,
            input_vocab_size, target_vocab_size, max_pos, dropout_rate
        )

    batch_size = 64
    epochs = 100
 

    # Slicing the dataset into encoder_input,decoder_input,decoder_target from x,y,y_trgx   
    dataset = tf.data.Dataset.from_tensor_slices((x, Y_input, Y_target))
    dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    best_loss = float('inf')
    # start with 7
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        total_loss = 0

        for batch, (encoder_input, decoder_input, decoder_target) in enumerate(dataset):
            batch_loss = train_step(encoder_input, decoder_input, decoder_target)
            total_loss += batch_loss

            if batch % 50 == 0:
                print(f"Batch {batch}, Loss: {batch_loss.numpy():.4f}")

        epoch_loss = total_loss.numpy() / (batch + 1)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

        # Save the model if this epoch has the best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            transformer.save('transformer_best_model.keras')
            transformer.save('transformer_best_model1.h5')
            print(f"Best model saved at epoch {epoch + 1} with loss {best_loss:.4f}")


