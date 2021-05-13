import torch

USE_CUDA = torch.cuda.is_available()
# USE_CUDA = False
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MAX_LENGTH = 35  # Maximum sentence length to consider
MIN_WORD_COUNT = 3
corpus_name = "medical_conversations"
dataset_path = '../data/healthtap_full_qa_processed_20k_words.csv'

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
# attn_model = 'concat'
hidden_size = 1024
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.3
batch_size = 128

# Set checkpoint to load from; set to None if starting from scratch
# loadFilename = "./data/save/cb_model/medical conversations/2-2_1024/30000_checkpoint.tar"
loadFilename = None
checkpoint_iter = 4000

clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 5000
n_val_iteration = 1000
n_epoch = 100
print_every = 100
save_every = 5000

train_time = 0
