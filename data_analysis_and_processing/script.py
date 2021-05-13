import pandas as pd
import progressbar
data = pd.read_csv("./healthtap_full_qa_processed_30k_words.csv")
data = data[['question', 'answer']]
data.dropna(inplace=True)
qa_pairs = data.values.tolist()
words = {}

def filter_words(text):
    new_sentence = ''
    for w in text.split(' '):
        if words[w] < 80:
            new_sentence += ':unk:' + ' '
        else:
            new_sentence += w + ' '

    return new_sentence[:-1]


bar = progressbar.ProgressBar(maxval=len(qa_pairs), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

for pair in qa_pairs:
    for word in pair[0].split(' '):
        if word not in words:
            words[word] = 1
        else:
            words[word] += 1
    for word in pair[1].split(' '):
        if word not in words:
            words[word] = 1
        else:
            words[word] += 1

qa_pairs_processed = []

bar.start()
for i, pair in enumerate(qa_pairs):
    qa_pairs_processed.append([filter_words(pair[0]), filter_words(pair[1])])
    bar.update(i+1)
bar.finish()

data = pd.DataFrame(qa_pairs_processed, columns=['question', 'answer'])

full_data_path = './healthtap_full_qa_processed_20k_words.csv'
data.to_csv(full_data_path)
