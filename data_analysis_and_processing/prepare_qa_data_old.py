import pandas as pd
import re
import unicodedata
import time
import numpy as np

URL_REGEX = "(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+"
EMAIL_REGEX = "[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*"


def phrase_expanding(text):
    text = re.sub(r"(\b)([Ii])['|’]m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)['|’]re", r"\1\2 are", text)
    text = re.sub(r"(\b)([Ll]et)['|’]s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)['|’]ll", r"\1\2 will", text)
    text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)['|’]ve", r"\1\2 have", text)

    return text

def word_negation(text):
    text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]as|[Ww]ould)n['|’]?t", r"\1\2 not", text)
    text = re.sub(r"(\b)([Cc]a)n['|’]t", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ww])on['|’]t", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han['|’]t", r"\1\2hall not", text)

    return text

def replace_non_utf(text):
    text = re.sub(r"Â", r"", text)
    text = re.sub(r"â€™", r"'", text)
    text = re.sub(r"â€œ", r'"', text)
    text = re.sub(r"â€“", r"-", text)
    text = re.sub(r"â€", r'"', text)

    return text

def fix_words(text):
    text = re.sub(r" drs ", r" doctors ", text)
    text = re.sub(r"^drs ", r"doctors ", text)
    text = re.sub(r" drs$", r" doctors", text)
    text = re.sub(r"yrs", r"years", text)

    return text

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(text):
    text = re.sub(URL_REGEX, ' :url: ', text)
    text = re.sub(EMAIL_REGEX, ' :email: ', text)

    text = unicode_to_ascii(text.lower().strip())
    text = re.sub(r"([.!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.!?:]+", r" ", text)
    text = re.sub(r"\s+", r" ", text).strip()
    return text

def preprocess_text(text):
    text = phrase_expanding(text)
    text = word_negation(text)
    text = replace_non_utf(text)
    text = normalize_string(text)
    text = fix_words(text)
    
    return text

pd.set_option('float_format', '{:f}'.format)

start = time.time()

data = pd.read_csv('./data/healthtap_full_qa.csv')
data = data[['question', 'answer']]

print('data loaded - ' + str(len(data)) + ' records')
print('Number of question NA values ' + str(len(data[data['question'].isna()])))
print('Number of answer NA values ' + str(len(data[data['answer'].isna()])))

# toto mi nefunguje neviem preco
data.dropna(inplace=True)

data['question_len'] = data['question'].str.split().apply(len)
data['answer_len'] = data['answer'].str.split().apply(len)
question_quantile = data['question_len'].quantile(0.95)
answer_quantile = data['answer_len'].quantile(0.95)
print('question 0.95 quantile ' + str(question_quantile))
print('answer 0.95 quantile ' + str(answer_quantile))

data = data[data['question_len'] < question_quantile]
data = data[data['answer_len'] < answer_quantile]

print(data['question_len'].describe())
print(data['answer_len'].describe())

print('starting preprocessing...')
# data['question'] = data.apply(lambda x: preprocess_text(x['question']), axis=1)
data['question'] = np.vectorize(preprocess_text)(data['question'])
# data['question'] = np.vectorize(preprocess_text)(data.to_numpy()[:,0])
print('questions preprocessed')
# data['answer'] = data.apply(lambda x: preprocess_text(x['answer']), axis=1)
data['answer'] = np.vectorize(preprocess_text)(data['answer'])
# data['answer'] = np.vectorize(preprocess_text)(data.to_numpy()[:,1])
print('answers preprocessed')

data = data[['question', 'answer']]
data = data.replace(to_replace = ' Do you want to video or text chat .*', value = '', regex = True)


full_data_path = './data/healthtap_full_qa_processed.csv'
data.to_csv(full_data_path)


print('FULL DATASET SAVED TO ' + full_data_path)

data_small_subset = data.sample(n=1000)
small_data_path = './data/healthtap_1000_qa_processed.csv'
data_small_subset.to_csv(small_data_path)

print('SMALL DATASET SAVED TO ' + small_data_path)

end = time.time()
print('Duration')
print(end - start)