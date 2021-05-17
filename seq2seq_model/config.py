import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

training = False
evaluation = True

PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 35
MIN_WORD_COUNT = 3

corpus_name = "medical_conversations"
dataset_path = './healthtap_full_qa_processed_30k_words.csv'

model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 1024
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 128

loadFilename = "./16_5000_checkpoint.tar"
# loadFilename = None
checkpoint_iter = 4000

clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 5000
n_val_iteration = 2
n_epoch = 100
print_every = 1
save_every = 5000

train_time = 0

