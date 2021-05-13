import torch

device = torch.device("cpu")
print(device)

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MAX_LENGTH = 35  # Maximum sentence length to consider
MIN_WORD_COUNT = 3
corpus_name = "medical conversations"
dataset_path = './healthtap_full_qa_processed_30k_words.csv'

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 1024
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = "./16_5000_checkpoint.tar"
# loadFilename = None
checkpoint_iter = 4000

clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 30000
n_epoch = 5
print_every = 1
save_every = 5000

encoder_time = 0
decoder_time = 0
train_time = 0
backward_time = 0
