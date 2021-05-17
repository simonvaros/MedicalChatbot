import pandas as pd
import re
import unicodedata
import time
import numpy as np
import math
from spellchecker import SpellChecker
import progressbar
from multiprocessing.dummy import Pool as ThreadPool
from threading import Thread


def main():
    URL_REGEX = "(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+"
    EMAIL_REGEX = "[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*"

    start = time.time()

    spell = SpellChecker()

    # global regex_ops_time
    regex_ops_time = 0
    # global unicode_to_ascii_time
    unicode_to_ascii_time = 0
    # global spell_check_time
    spell_check_time = 0

    def phrase_expanding(text):
        s = time.time()

        text = re.sub(r"(\b)([Ii])['|’]m", r"\1\2 am", text)
        text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)['|’]re", r"\1\2 are", text)
        text = re.sub(r"(\b)([Ll]et)['|’]s", r"\1\2 us", text)
        text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)['|’]ll", r"\1\2 will", text)
        text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)['|’]ve", r"\1\2 have", text)

        e = time.time()
        nonlocal regex_ops_time
        regex_ops_time += e - s

        return text


    def word_negation(text):
        s = time.time()

        text = re.sub(
            r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]as|[Ww]ould)n['|’]?t",
            r"\1\2 not", text)
        text = re.sub(r"(\b)([Cc]a)n['|’]t", r"\1\2n not", text)
        text = re.sub(r"(\b)([Ww])on['|’]t", r"\1\2ill not", text)
        text = re.sub(r"(\b)([Ss])han['|’]t", r"\1\2hall not", text)

        e = time.time()
        nonlocal regex_ops_time
        regex_ops_time += e - s

        return text


    def replace_non_utf(text):
        s = time.time()

        text = re.sub(r"Â", r"", text)
        text = re.sub(r"â€™", r"'", text)
        text = re.sub(r"â€œ", r'"', text)
        text = re.sub(r"â€“", r"-", text)
        text = re.sub(r"â€", r'"', text)

        e = time.time()
        nonlocal regex_ops_time
        regex_ops_time += e - s

        return text


    def fix_words(text):
        s = time.time()

        text = re.sub(r" drs ", r" doctors ", text)
        text = re.sub(r"^drs ", r"doctors ", text)
        text = re.sub(r" drs$", r" doctors", text)
        text = re.sub(r"yrs", r"years", text)

        e = time.time()
        nonlocal regex_ops_time
        regex_ops_time += e - s

        return text


    def unicode_to_ascii(text):
        s = time.time()

        res = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

        e = time.time()
        nonlocal unicode_to_ascii_time
        unicode_to_ascii_time += e - s

        return res


    # Lowercase, trim, and remove non-letter characters
    def normalize_string(text):
        s = time.time()

        text = re.sub(URL_REGEX, ' :url: ', text)
        text = re.sub(EMAIL_REGEX, ' :email: ', text)

        e = time.time()
        nonlocal regex_ops_time
        regex_ops_time += e - s

        text = unicode_to_ascii(text.lower().strip())

        s = time.time()

        text = re.sub(r"([.!?])", r" \1 ", text)
        text = re.sub(r"[^a-zA-Z.!?:]+", r" ", text)
        text = re.sub(r"\s+", r" ", text).strip()

        e = time.time()
        regex_ops_time += e - s

        return text

    def spell_check(text, words):
        s = time.time()

        sentence = ''

        for word in text.split(' '):
            res = word

            if words[word] < 5:
                res = spell.correction(word)

            sentence += res + ' '

        e = time.time()
        nonlocal spell_check_time
        spell_check_time += e - s

        return sentence[:-1]

    def preprocess_text(text):
        text = phrase_expanding(text)
        text = word_negation(text)
        text = replace_non_utf(text)
        text = normalize_string(text)
        text = fix_words(text)

        return text


    pd.set_option('float_format', '{:f}'.format)

    data = pd.read_csv('./healthtap_full_qa.csv')
    data = data[['question', 'answer']]

    print('data loaded - ' + str(len(data)) + ' records')


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


    data = data.replace(to_replace=' Do you want to video or text chat .*', value='', regex=True)


    qa_pairs = data.values.tolist()
    qa_pairs_processed = []

    words = {}


    print('starting preprocessing...')

    bar = progressbar.ProgressBar(maxval=len(qa_pairs), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    bar.start()
    for i, pair in enumerate(qa_pairs):
        qa_pairs_processed.append([preprocess_text(pair[0]), preprocess_text(pair[1])])
        bar.update(i+1)
    bar.finish()

    print('preprocessing done')
    print('starting generating vocab')

    for pair in qa_pairs_processed:
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

    words = dict(sorted(words.items(), key=lambda item: item[1]))

    print('vocab done')

    qa_pairs_corrected = qa_pairs_processed

    bar.start()
    for i, pair in enumerate(qa_pairs_processed):
        qa_pairs_corrected.append([spell_check(pair[0], words), spell_check(pair[1], words)])
        bar.update(i+1)
    bar.finish()

    print('Number of question NA values ' + str(len(data[data['question'].isna()])))
    print('Number of answer NA values ' + str(len(data[data['answer'].isna()])))

    data.dropna(inplace=True)

    data = pd.DataFrame(qa_pairs_corrected, columns=['question', 'answer'])

    # full_data_path = './healthtap_full_qa_processed.csv'
    data.to_csv(full_data_path)

    print('FULL DATASET SAVED TO ' + full_data_path)

    # data_small_subset = data.sample(n=1000)
    # small_data_path = './healthtap_1000_qa_processed.csv'
    # data_small_subset.to_csv(small_data_path)
    #
    # print('SMALL DATASET SAVED TO ' + small_data_path)

    end = time.time()
    print('Duration')
    print(end - start)

    print(regex_ops_time)
    print(unicode_to_ascii_time)
    print(spell_check_time)

main()