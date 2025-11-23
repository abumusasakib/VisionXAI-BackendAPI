import os
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io
import base64
from PIL import Image
import math
from loguru import logger

# Desired image dimensions (match InceptionV3)
IMAGE_SIZE = (299, 299)

# Vocabulary size
VOCAB_SIZE = 6000

# Fixed length allowed for any sequence
SEQ_LENGTH = 8

# Dimension for the image embeddings and token embeddings
EMBED_DIM = 256

# Per-layer units in the feed-forward network
FF_DIM = 128

# Model Version
mdx = "20250714_141134"

# Directory Path
WEIGHTS_DIR = "ImgCap/weights/"
# Determine which vocab/tokenizer file to use. Prefer tokenizer.pkl if present.
DEFAULT_VOCAB_FILENAME = f"vocab_{mdx}"
TOKENIZER_PKL = os.path.join(WEIGHTS_DIR, "tokenizer.pkl")
VOCAB_FILE = os.path.join(WEIGHTS_DIR, DEFAULT_VOCAB_FILENAME)
if os.path.exists(TOKENIZER_PKL):
    VOCAB_FILE = TOKENIZER_PKL


def load_vocab(filepath):
    try:
        # If this is a keras Tokenizer pickle, unpickling requires keras_preprocessing to be importable.
        if os.path.basename(filepath).lower().startswith("tokenizer"):
            try:
                import keras_preprocessing  # noqa: F401
            except Exception:
                logger.warning(
                    "Detected tokenizer.pkl but keras_preprocessing is not available; skipping tokenizer load.\n"
                    "Install 'keras_preprocessing' in your environment to enable tokenizer unpickling."
                )
                return None
        logger.info(f"Loading vocabulary/tokenizer from {filepath}")
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        logger.info("Vocabulary/tokenizer loaded successfully")

        # Determine object type
        if hasattr(obj, "texts_to_sequences"):
            # It's a Tokenizer
            logger.info("Loaded object is a Tokenizer")
            return obj
        elif isinstance(obj, (list, tuple)):
            # Vocabulary list
            return list(obj)
        elif isinstance(obj, dict):
            # dictionary mapping (e.g., index->token or token->index)
            try:
                keys = sorted(obj.keys())
                # if keys are integers -> build index_lookup
                if all(isinstance(k, int) for k in keys):
                    vocab_list = [obj[k] for k in keys]
                    return vocab_list
            except Exception:
                pass
            # fallback: return keys as vocab
            return list(obj)
        else:
            # unknown type, return as-is
            return obj
    except Exception as e:
        logger.error(f"Error loading vocabulary file: {e}")
        return None


raw_vocab_obj = load_vocab(VOCAB_FILE)

# Prepare tokenizer/vectorization/index lookup depending on the loaded object
tokenizer = None
vectorization = None
index_lookup = None
if raw_vocab_obj is None:
    logger.warning("No vocabulary/tokenizer loaded. Caption generation may not work.")
else:
    # If it's a Tokenizer instance
    if hasattr(raw_vocab_obj, "texts_to_sequences"):
        tokenizer = raw_vocab_obj
        # word_index maps token->index; build index_lookup (index->token)
        try:
            word_index = tokenizer.word_index
            # Keras tokenizer indices start at 1; build index->word mapping
            index_lookup = {idx: word for word, idx in word_index.items()}
            VOCAB_SIZE = max(word_index.values()) + 1
            logger.info(f"Tokenizer loaded: VOCAB_SIZE={VOCAB_SIZE}")
        except Exception as e:
            logger.error(f"Failed to initialize index lookup from tokenizer: {e}")
    else:
        # Assume it's a plain vocabulary list
        vocab = list(raw_vocab_obj)

        # Custom standardization for TextVectorization
        def custom_standardization(input_string):
            lowercase = tf.strings.lower(input_string)
            return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

        # Initialize TextVectorization
        try:
            vectorization = TextVectorization(
                max_tokens=VOCAB_SIZE,
                output_mode="int",
                output_sequence_length=SEQ_LENGTH,
                standardize=custom_standardization,
                vocabulary=vocab,
            )
            index_lookup = dict(zip(range(len(vocab)), vocab))
            VOCAB_SIZE = len(vocab)
            logger.info(f"TextVectorization initialized with VOCAB_SIZE={VOCAB_SIZE}")
        except Exception as e:
            logger.error(f"Error initializing TextVectorization: {e}")

# Data augmentation for image data
image_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),  # Reduced rotation for faster preprocessing
        layers.RandomContrast(0.2),  # Lighter contrast adjustment
        # layers.RandomTranslation(0.1, 0.1),
    ]
)


# Decode, resize, and preprocess images
def decode_and_resize(img_path):
    """Read image from disk, resize and apply InceptionV3 preprocess_input."""
    try:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return img
    except Exception as e:
        logger.error(f"❌ Error in decoding and resizing image {img_path}: {e}")


def decode_and_resize_bytes(image_bytes):
    """Decode image bytes (from UploadFile) and preprocess for InceptionV3."""
    try:
        img = tf.image.decode_jpeg(image_bytes, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return img
    except Exception as e:
        logger.error(f"❌ Error decoding image bytes: {e}")
        return None


# Defining the Model
# CNN
def get_cnn_model():
    try:
        base_model = InceptionV3(
            input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
        )
        base_model.trainable = False

        # Output is (None, 8, 8, 2048) for InceptionV3
        base_output = base_model.output

        # Flatten spatial dimensions into sequence: (None, 64, 2048)
        # NOTE: Do NOT pre-project to EMBED_DIM here - the checkpoint's
        # encoder expects raw Inception features of size 2048 and will
        # perform the projection (encoder.fc). Keeping projection here
        # causes a shape mismatch with checkpointed variables.
        seq = layers.Reshape((-1, int(base_output.shape[-1])))(base_output)

        cnn_model = keras.models.Model(base_model.input, seq)
        logger.info("InceptionV3-based CNN feature extractor loaded")
        return cnn_model
    except Exception as e:
        logger.error(f"Error loading CNN model: {e}")


# Attention and RNN-based encoder/decoder


class BahdanauAttention(keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features, hidden):
        # features: (batch_size, 64, embedding_dim)
        # hidden: (batch_size, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)  # (batch_size, 1, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        score = self.V(score)  # (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch_size, 64, 1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch_size, hidden_size)
        return context_vector, attention_weights


class CNN_Encoder(keras.Model):
    """Simple encoder that projects Inception features to embedding_dim.
    """

    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.gru = layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fc1 = layers.Dense(self.units)
        self.fc2 = layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # x: (batch_size, 1) token ids for the current step
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)  # (batch_size, 1, embedding_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)  # (batch,1, embedding+hidden)
        output, state = self.gru(x)
        x = self.fc1(output)  # (batch,1, units)
        x = tf.reshape(x, (-1, x.shape[2]))  # (batch, units)
        x = self.fc2(x)  # (batch, vocab_size)
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


# Model definition
class ImageCaptioningModel(keras.Model):
    def __init__(
        self,
        cnn_model,
        encoder,
        decoder,
        num_captions_per_image=5,
        image_aug=None,
    ):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        # Use sparse categorical crossentropy (from_logits=True)
        self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        self.num_captions_per_image = num_captions_per_image
        self.image_aug = image_aug

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
        # logger.debug(
        #     f"Image Embedding Input Shape before passing to Encoder: {img_embed.shape}"
        # )

        # batch_seq = tf.expand_dims(batch_seq, axis=1)
        # logger.debug(f"Batch Sequence Input Shape before slicing: {batch_seq.shape}")

        # Run encoder (CNN_Encoder) to get image features for attention
        encoder_out = self.encoder(img_embed)

        # Prepare sequences
        batch_seq_inp = batch_seq[:, :-1]  # Input sequence (without the last token)
        batch_seq_true = batch_seq[:, 1:]  # Target sequence (without the first token)
        mask = tf.math.not_equal(batch_seq_true, 0)

        batch_size = tf.shape(batch_seq)[0]
        seq_len = tf.shape(batch_seq_inp)[1]

        # Initialize decoder hidden state
        hidden = self.decoder.reset_state(batch_size)

        total_loss = tf.constant(0.0, dtype=tf.float32)
        total_acc = tf.constant(0.0, dtype=tf.float32)
        total_mask = tf.reduce_sum(tf.cast(mask, tf.float32))

        # Iterate over time steps (teacher forcing)
        for t in range(int(SEQ_LENGTH) - 1):
            # current input token (batch,)
            x_t = tf.expand_dims(batch_seq_inp[:, t], 1)  # (batch,1)
            preds, hidden, _ = self.decoder(x_t, encoder_out, hidden)
            # preds: (batch, vocab)
            y_true = batch_seq_true[:, t]
            mask_t = tf.cast(mask[:, t], dtype=tf.float32)

            # Compute loss per example and apply mask
            loss_t = self.loss(y_true, preds)  # (batch,)
            loss_t = loss_t * mask_t
            total_loss += tf.reduce_sum(loss_t)

            # Compute accuracy for this step
            preds_id = tf.argmax(preds, axis=1, output_type=tf.int32)
            acc_t = tf.cast(tf.equal(y_true, preds_id), tf.float32) * mask_t
            total_acc += tf.reduce_sum(acc_t)

        # Prevent division by zero
        denom = tf.maximum(total_mask, 1.0)
        avg_loss = total_loss / denom
        avg_acc = total_acc / denom
        return avg_loss, avg_acc

    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data

        # batch_seq = tf.expand_dims(batch_seq, axis=1)

        # logger.debug(
        #     f"Training Image Batch Shape before passing to CNN: {batch_img.shape}"
        # )
        total_loss = 0
        total_acc = 0

        if self.image_aug:
            batch_img = self.image_aug(batch_img)

        # logger.debug(f"Training Image Batch Shape: {batch_img.shape}")
        # logger.debug(f"Training Sequence Batch Shape: {batch_seq.shape}")

        # 1. Get image embeddings from CNN
        img_embed = self.cnn_model(batch_img)
        # logger.debug(f"Image Embeddings Shape: {img_embed.shape}")

        # 2. Ensure CNN output has a sequence dimension: if cnn returns (batch, embed_dim)
        # expand to (batch, 1, embed_dim). If it already returns (batch, seq_len, embed_dim)
        # keep as-is.
        if img_embed.shape.ndims == 2:
            img_embed = tf.expand_dims(img_embed, axis=1)

        # logger.debug(f"Reshaped Image Embeddings for Encoder: {img_embed.shape}")

        # 3. Make sure batch_seq has 3 dimensions
        if batch_seq.shape.ndims == 2:
            # Reshape the sequence to have a third dimension (e.g., 1 caption per image)
            batch_seq = tf.expand_dims(batch_seq, axis=1)

        # logger.debug(f"Updated Sequence Shape: {batch_seq.shape}")

        # 4. Accumulate loss and accuracy for each caption
        with tf.GradientTape() as tape:
            # Loop through each caption (batch_seq should be (batch_size, num_captions, sequence_length))
            num_captions_per_image = batch_seq.shape[
                1
            ]  # Extract the num_captions dimension

            for i in range(self.num_captions_per_image):
                loss, acc = self._compute_caption_loss_and_acc(
                    img_embed, batch_seq[:, i, :], training=True
                )
                total_loss += loss
                total_acc += acc

            # 5. Compute the mean loss and accuracy
            avg_loss = total_loss / tf.cast(
                self.num_captions_per_image, dtype=tf.float32
            )
            avg_acc = total_acc / tf.cast(self.num_captions_per_image, dtype=tf.float32)

        # Backpropagation
        # 6. Get the list of all the trainable weights
        train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables

        # 7. Get the gradients (from the accumulated loss)
        grads = tape.gradient(avg_loss, train_vars)

        # 8. Update the trainable weights
        self.optimizer.apply_gradients(zip(grads, train_vars))

        # 9. Update the trackers
        self.loss_tracker.update_state(avg_loss)
        self.acc_tracker.update_state(avg_acc)

        # 10. Return the loss and accuracy values
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        # logger.debug(f"Validation Image Batch Shape: {batch_img.shape}")
        # logger.debug(f"Validation Sequence Batch Shape: {batch_seq.shape}")

        # batch_seq = tf.expand_dims(batch_seq, axis=1)

        batch_loss = 0
        batch_acc = 0

        # 1. Get image embeddings
        img_embed = self.cnn_model(batch_img)
        logger.debug(f"Image Embeddings Shape: {img_embed.shape}")
        if img_embed.shape.ndims == 2:
            img_embed = tf.expand_dims(img_embed, axis=1)
        logger.debug(f"Reshaped Image Embeddings Shape: {img_embed.shape}")

        # 2. Pass each of the captions one by one to the decoder
        # along with the encoder outputs and compute the loss as well as accuracy
        # for each caption.
        # Loop through captions
        for i in range(self.num_captions_per_image):
            batch_seq_inp = batch_seq[:, i, :-1]
            batch_seq_true = batch_seq[:, i, 1:]
            # logger.debug(f"Validation Sequence Input Shape: {batch_seq_inp.shape}")
            # logger.debug(f"Validation Sequence True Shape: {batch_seq_true.shape}")

            loss, acc = self._compute_caption_loss_and_acc(
                img_embed, batch_seq[:, i, :], training=False
            )

            # 3. Update batch loss and batch accuracy
            batch_loss += loss
            batch_acc += acc

        batch_acc /= float(self.num_captions_per_image)

        # 4. Update the trackers
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # 5. Return the loss and accuracy values
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]


# Model construction

# Initialize components
cnn_model = get_cnn_model()
encoder = CNN_Encoder(EMBED_DIM)
# Detect vocab size from checkpoint (if available) to avoid embedding shape mismatches.
# Prefer checkpoint vocab size over tokenizer initial VOCAB_SIZE so decoder variables
# align with checkpointed embedding shapes.
try:
    ckpt_dir = os.path.join(WEIGHTS_DIR, "best_model")
    ckpt_prefix = tf.train.latest_checkpoint(ckpt_dir) or tf.train.latest_checkpoint(WEIGHTS_DIR)
    if ckpt_prefix:
        try:
            ck = tf.train.load_checkpoint(ckpt_prefix)
            vars_map = ck.get_variable_to_shape_map()
            for nm, shp in vars_map.items():
                if "decoder/embedding/embeddings" in nm:
                    try:
                        inferred_vocab = int(shp[0])
                        VOCAB_SIZE = inferred_vocab
                        logger.info(f"Inferred VOCAB_SIZE from checkpoint: {VOCAB_SIZE}")
                    except Exception:
                        pass
                    break
        except Exception:
            pass
except Exception:
    pass

# Use 512 units for the GRU decoder
decoder = RNN_Decoder(embedding_dim=EMBED_DIM, units=512, vocab_size=VOCAB_SIZE)

# Create the ImageCaptioningModel
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model,
    encoder=encoder,
    decoder=decoder,
    image_aug=image_augmentation,
)


# Load weights
def load_weights(filepath):
    # Check for the files
    try:
        # Loading weights: support several checkpoint naming conventions.
        fls = os.listdir(WEIGHTS_DIR)

        # imgcap_<mdx> checkpoints
        checkpoint_files = [f for f in fls if "imgcap_" in f]
        if len(checkpoint_files) > 0:
            logger.info("Found imgcap_* checkpoint files, attempting to load weights...")
            caption_model.load_weights(filepath)
            logger.info("Saved imgcap_* weights loaded successfully")
            return

        # Check for a best_model folder with TensorFlow checkpoints
        best_dir = os.path.join(WEIGHTS_DIR, "best_model")
        if os.path.isdir(best_dir):
            ckpt = tf.train.latest_checkpoint(best_dir)
            if ckpt:
                logger.info(f"Found checkpoint in best_model: {ckpt}, loading...")
                try:
                    caption_model.load_weights(ckpt)
                    logger.info("Checkpoint from best_model loaded successfully via Keras load_weights")
                    return
                except Exception as e:
                    logger.error(f"Error loading weights (keras load_weights): {e}")
                    logger.info("Attempting to restore variables via tf.train.Checkpoint")
                    try:
                        ck = tf.train.Checkpoint(model=caption_model)
                        ck.restore(ckpt).expect_partial()
                        logger.info("Restored checkpoint using tf.train.Checkpoint (expect_partial used)")
                        return
                    except Exception as e2:
                        logger.error(f"tf.train.Checkpoint restore failed: {e2}")

        # Try latest checkpoint in weights root
        latest = tf.train.latest_checkpoint(WEIGHTS_DIR)
        if latest:
            logger.info(f"Found latest checkpoint in weights dir: {latest}, loading...")
            caption_model.load_weights(latest)
            logger.info("Latest checkpoint loaded successfully")
            return

        logger.info("No compatible checkpoint files found in weights directory; skipping weight load.")
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"Error loading weights: {str(e)}")


# Load weights
WEIGHTS_FILE = f"{WEIGHTS_DIR}imgcap_{mdx}"
load_weights(WEIGHTS_FILE)

# Initialize Vocabulary
if tokenizer is not None:
    try:
        # tokenizer.index_word maps indices to words
        if hasattr(tokenizer, "index_word"):
            index_lookup = tokenizer.index_word
        else:
            # build from word_index
            index_lookup = {v: k for k, v in tokenizer.word_index.items()}
        # update VOCAB_SIZE
        if hasattr(tokenizer, "word_index"):
            VOCAB_SIZE = max(tokenizer.word_index.values()) + 1
        logger.info(f"Tokenizer initialized: VOCAB_SIZE={VOCAB_SIZE}")
    except Exception as e:
        index_lookup = None
        logger.error(f"Failed to initialize tokenizer mappings: {e}")
elif vectorization is not None:
    try:
        vocab = vectorization.get_vocabulary()
        if vocab:
            index_lookup = dict(zip(range(len(vocab)), vocab))
            VOCAB_SIZE = len(vocab)
            logger.info(f"TextVectorization vocabulary size: {len(vocab)}")
        else:
            index_lookup = None
            logger.warning("TextVectorization vocabulary is empty.")
    except Exception as e:
        index_lookup = None
        logger.error(f"Failed to retrieve TextVectorization vocabulary: {e}")
else:
    index_lookup = None
    logger.warning("No tokenizer or vectorization available. Captions may not generate correctly.")

max_decoded_sentence_length = SEQ_LENGTH - 1


# Generate Caption
def generate(img_path):
    """Generate a caption for a local image path or return an error string."""
    try:
        # Decode and resize the image (InceptionV3 preprocessing)
        sample_img = decode_and_resize(img_path)
        if sample_img is None:
            logger.error("Image could not be processed.")
            return "Image could not be processed."

        # Create batch dim and extract features (cnn_model returns a sequence)
        img_tensor = tf.expand_dims(sample_img, 0)
        img_features = caption_model.cnn_model(img_tensor)  # (1, seq_len, EMBED_DIM)

        # Encode the image features with CNN_Encoder
        encoded_img = caption_model.encoder(img_features)

        # Initialize decoder state
        hidden = caption_model.decoder.reset_state(batch_size=1)

        # Determine start token id
        start_id = 1
        if tokenizer is not None and hasattr(tokenizer, "word_index"):
            start_id = tokenizer.word_index.get("<start>", tokenizer.word_index.get("start", 1))

        decoded_tokens = []
        input_id = start_id

        def _pick_token(probs, prev_id=None):
            """Pick a token index from probs while avoiding repeating prev_id if possible.
            If prev_id is the argmax, choose the next-best token instead. Falls back to argmax
            when masking would zero all probabilities.
            """
            probs = np.asarray(probs, dtype=float).copy()
            if prev_id is not None and 0 <= int(prev_id) < probs.size:
                # If the previous token has the highest probability, zero it and pick the next.
                max_idx = int(np.argmax(probs))
                if max_idx == int(prev_id):
                    probs[int(prev_id)] = 0.0
                    if probs.sum() > 0:
                        return int(np.argmax(probs))
                    # else fall through to return the original max
            return int(np.argmax(probs))

        for _ in range(max_decoded_sentence_length):
            input_tensor = tf.constant([[int(input_id)]], dtype=tf.int32)
            preds, hidden, _ = caption_model.decoder(input_tensor, encoded_img, hidden)
            probs = tf.nn.softmax(preds, axis=-1).numpy()[0]
            sampled_token_index = _pick_token(probs, prev_id=input_id)

            if index_lookup is None:
                logger.warning("Index lookup missing; cannot map token index to word.")
                break

            sampled_token = index_lookup.get(sampled_token_index, None)
            if sampled_token is None:
                logger.warning(f"Token index {sampled_token_index} not found in index_lookup")
                break
            # Filter out unknown / pad tokens (case-insensitive variants)
            sval = str(sampled_token).strip()
            if sval == "" or sval.lower() in ("[unk]", "<unk>", "[pad]", "<pad>"):
                input_id = sampled_token_index
                continue
            if sampled_token == "<end>":
                break
            decoded_tokens.append(sampled_token)
            input_id = sampled_token_index

        decoded_caption = " ".join(decoded_tokens).strip()
        logger.info(f"Generated caption for image {img_path}: {decoded_caption}")
        return decoded_caption
    except Exception as e:
        logger.error(f"Error generating caption for image {img_path}: {e}")
        return "Error generating caption."


def generate_from_bytes(image_bytes):
    """Generate caption from raw image bytes (e.g., UploadFile.read())."""
    try:
        # Preserve a PIL copy of the original image for attention plotting
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            pil_img = None

        img = decode_and_resize_bytes(image_bytes)
        if img is None:
            return "Image could not be processed."

        img_tensor = tf.expand_dims(img, 0)
        img_features = caption_model.cnn_model(img_tensor)
        encoded_img = caption_model.encoder(img_features)

        # Reuse generate logic by temporarily writing the encoded image path approach is avoided;
        # Instead duplicate minimal generation loop here.
        decoded_caption = ["<start>"]
        token_ids = []
        tokens = []
        token_scores = []
        attention_maps = []

        # Use RNN decoder step-wise and collect attention weights per step
        hidden = caption_model.decoder.reset_state(batch_size=1)

        # Determine start token id
        start_id = 1
        if tokenizer is not None and hasattr(tokenizer, "word_index"):
            start_id = tokenizer.word_index.get("<start>", tokenizer.word_index.get("start", 1))

        input_id = start_id
        def _pick_token(probs, prev_id=None):
            probs = np.asarray(probs, dtype=float).copy()
            if prev_id is not None and 0 <= int(prev_id) < probs.size:
                max_idx = int(np.argmax(probs))
                if max_idx == int(prev_id):
                    probs[int(prev_id)] = 0.0
                    if probs.sum() > 0:
                        return int(np.argmax(probs))
            return int(np.argmax(probs))

        for _ in range(max_decoded_sentence_length):
            input_tensor = tf.constant([[int(input_id)]], dtype=tf.int32)
            preds, hidden, attention_weights = caption_model.decoder(input_tensor, encoded_img, hidden)
            probs = tf.nn.softmax(preds, axis=-1).numpy()[0]
            sampled_token_index = _pick_token(probs, prev_id=input_id)
            sampled_score = float(probs[sampled_token_index])

            # attention_weights: (batch, 64, 1)
            att_for_step = None
            try:
                att_np = np.array(attention_weights)
                # squeeze last dim -> (batch, key_len)
                att_for_step = att_np[0, :, 0]
            except Exception:
                att_for_step = None

            if index_lookup is None:
                break
            sampled_token = index_lookup.get(sampled_token_index, None)
            if sampled_token is None:
                input_id = sampled_token_index
                continue
            sval = str(sampled_token).strip()
            if sval == "" or sval.lower() in ("[unk]", "<unk>", "[pad]", "<pad>"):
                input_id = sampled_token_index
                continue
            if sampled_token == "<end>":
                break
            decoded_caption.append(sampled_token)
            token_ids.append(sampled_token_index)
            tokens.append(sampled_token)
            token_scores.append(sampled_score)
            attention_maps.append(att_for_step)
            input_id = sampled_token_index

        caption = " ".join(decoded_caption[1:]).strip()

        # Build attention metrics and a notebook-style multi-panel attention image.
        attention_image_b64 = None
        attention_means = []
        attention_topk = []
        try:
            # Compute per-token attention metrics (mean and top-k grid locations)
            for a in attention_maps:
                if a is None:
                    attention_means.append(None)
                    attention_topk.append([])
                    continue
                arr = np.array(a)
                norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
                attention_means.append(float(norm.mean()))

                k = min(5, norm.size)
                top_idx = np.argsort(-norm)[:k]
                key_len = norm.size
                side = int(math.sqrt(key_len))
                topk_list = []
                for flat in top_idx:
                    score = float(norm[flat])
                    if side * side == key_len:
                        r = int(flat // side)
                        c = int(flat % side)
                    else:
                        r = 0
                        c = int(flat)
                    topk_list.append((r, c, score))
                attention_topk.append(topk_list)

            # Create a single overlay figure using matplotlib if available. The overlay
            # composites each token's attention map in a different color (no labels/titles).
            if pil_img is not None and any(m is not None for m in attention_maps):
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.cm as cm

                    # Prepare base image as float RGB (0..1)
                    base_rgb = np.array(pil_img).astype(np.float32) / 255.0

                    # Find grid shape from the first non-none attention map
                    grid_rows = None
                    grid_cols = None
                    for a in attention_maps:
                        if a is None:
                            continue
                        key_len = int(a.shape[0])
                        side = int(math.sqrt(key_len))
                        if side * side == key_len:
                            grid_rows = side
                            grid_cols = side
                        else:
                            grid_rows = 1
                            grid_cols = key_len
                        break

                    # Composite overlays sequentially using distinct colors from a qualitative cmap
                    cmap = matplotlib.cm.get_cmap("tab20")
                    composite = base_rgb.copy()

                    # Strength factor controls how strong each overlay is; smaller avoids saturation
                    overlay_strength = 0.55

                    for idx, att in enumerate(attention_maps):
                        if att is None:
                            continue
                        key_len = int(att.shape[0])
                        side = int(math.sqrt(key_len))
                        if side * side == key_len:
                            att_grid = att.reshape((side, side))
                        else:
                            att_grid = att.reshape((1, key_len))

                        # Normalize attention map to 0..1
                        att_norm = (att_grid - att_grid.min()) / (att_grid.max() - att_grid.min() + 1e-9)
                        att_img = Image.fromarray(np.uint8(255 * att_norm)).resize(pil_img.size, Image.BILINEAR)
                        att_arr = np.array(att_img).astype(np.float32) / 255.0

                        # Pick a color for this token from the colormap
                        color = np.array(cmap(idx % cmap.N)[:3], dtype=np.float32)

                        # Create overlay RGB image (H,W,3) = att_arr[...,None] * color
                        overlay_rgb = np.dstack([att_arr * c for c in color])
                        alpha_map = att_arr * overlay_strength

                        # Alpha composite: composite = composite*(1-alpha) + overlay_rgb*alpha
                        alpha_exp = np.expand_dims(alpha_map, axis=2)
                        composite = composite * (1.0 - alpha_exp) + overlay_rgb * alpha_exp

                    # Clip and convert back to uint8
                    composite = np.clip(composite * 255.0, 0, 255).astype(np.uint8)
                    out_img = Image.fromarray(composite)

                    buf = io.BytesIO()
                    out_img.save(buf, format="PNG")
                    buf.seek(0)
                    attention_image_b64 = base64.b64encode(buf.read()).decode("utf-8")

                    # Store grid metadata (rows, cols) for client grid override
                    if grid_rows is None or grid_cols is None:
                        attention_grid = None
                        attention_shape = None
                    else:
                        attention_grid = [int(grid_rows), int(grid_cols)]
                        attention_shape = {"rows": int(grid_rows), "cols": int(grid_cols)}

                except Exception as e:
                    logger.debug(f"Matplotlib plotting failed, falling back to PIL overlay: {e}")
                    # Fallback behavior: create a single overlay from the last available attention map
                    last_att = None
                    for a in reversed(attention_maps):
                        if a is not None:
                            last_att = a
                            break
                    if last_att is not None:
                        key_len = last_att.shape[0]
                        side = int(math.sqrt(key_len))
                        if side * side == key_len:
                            att_grid = last_att.reshape((side, side))
                            attention_grid = [side, side]
                            attention_shape = {"rows": side, "cols": side}
                        else:
                            att_grid = last_att.reshape((1, key_len))
                            attention_grid = [1, key_len]
                            attention_shape = {"rows": 1, "cols": key_len}

                        att_norm = (att_grid - att_grid.min()) / (att_grid.max() - att_grid.min() + 1e-9)
                        att_img = Image.fromarray(np.uint8(255 * att_norm)).resize(pil_img.size, Image.BILINEAR).convert("L")
                        overlay = Image.new("RGBA", pil_img.size, (255, 0, 0, 0))
                        alpha = Image.fromarray(np.uint8(190 * (np.array(att_img) / 255.0))).convert("L")
                        overlay.putalpha(alpha)
                        base = pil_img.convert("RGBA")
                        composed = Image.alpha_composite(base, overlay)
                        buf = io.BytesIO()
                        composed.save(buf, format="PNG")
                        buf.seek(0)
                        attention_image_b64 = base64.b64encode(buf.read()).decode("utf-8")
                    else:
                        attention_grid = None
                        attention_shape = None
        except Exception as e:
            logger.error(f"Failed to build attention image or metrics: {e}")

        # Convert attention_topk tuples into structured dicts for client consumption
        # Also ensure any unknown/pad tokens are filtered out from attention outputs
        unknown_set = {"[unk]", "<unk>", "[pad]", "<pad>"}

        # Filter the raw attention_topk to align with the accepted `tokens` list
        # (safety in case any mismatch occurred during generation)
        filtered_attention_topk = []
        for tok, topk_lst in zip(tokens, attention_topk):
            sval = str(tok).strip().lower()
            if sval == "" or sval in unknown_set:
                # skip entries corresponding to unknown/pad tokens
                continue
            filtered_attention_topk.append(topk_lst)
        attention_topk = filtered_attention_topk

        attention_topk_items = []
        for tok, lst in zip(tokens, attention_topk):
            sval = str(tok).strip().lower()
            if sval == "" or sval in unknown_set:
                continue
            item_list = []
            for entry in lst:
                try:
                    r, c, sc = entry
                    item_list.append({"row": int(r), "col": int(c), "score": float(sc)})
                except Exception:
                    # If entry already dict-like or malformed, try to coerce
                    try:
                        item_list.append({"row": int(entry[0]), "col": int(entry[1]), "score": float(entry[2])})
                    except Exception:
                        continue
            attention_topk_items.append(item_list)

        # Prepare attention_image_bytes (base64) for clients that expect a key named _bytes
        attention_image_bytes_b64 = attention_image_b64

        # Return structured output including attention metrics
        return {
            "caption": caption,
            "token_ids": token_ids,
            "tokens": tokens,
            "token_scores": token_scores,
            "attention_image": attention_image_b64,
            "attention_image_bytes": attention_image_bytes_b64,
            "attention_means": attention_means,
            "attention_topk": attention_topk,
            "attention_topk_items": attention_topk_items,
            "attention_grid": attention_grid if 'attention_grid' in locals() else None,
            "attention_shape": attention_shape if 'attention_shape' in locals() else None,
        }
    except Exception as e:
        logger.error(f"Error generating caption from bytes: {e}")
        return "Error generating caption."
