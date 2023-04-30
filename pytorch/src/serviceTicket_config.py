import transformers

MAX_LEN = 128 # Bytes to feed into BERT Model
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10 # Number of Training Iterations
BASE_MODEL_PATH = "bert-base-uncased" # See documentation at https://huggingface.co/bert-base-uncased
MODEL_PATH = "model.bin" # Output model structure
TRAINING_FILE = "../input/input_training_data.csv" # Training Data created from serviceTicket_format_rawData

TOKENIZER = transformers.BertTokenizer.from_pretrained(BASE_MODEL_PATH)