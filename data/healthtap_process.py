import pandas as pd
import time
import sys

def transformAnswer(a, s):
    try:
        a['main_category'] = s.main_category
        a['question'] = s.question
        a['related_topics'] = s.related_topics
        a['sub_category'] = s.sub_category
    except:
        return

    global my_dict
    global i
    my_dict[i] = a
    i = i + 1

def transformRow(s):
    try:
        answers = eval(s.answers) 
        for item in answers:
            transformAnswer(item, s)

        global count
        count = count + 1

        if count % 10000 == 0:
            global start
            global i
            endTmp = time.time()
            print('Processed {} after {} seconds. processed data length = {}'.format(count, endTmp - start, i))
    except:
        return

data = pd.read_csv('../../data/healthtap_full.csv')
data = data.drop(columns=['Unnamed: 0'])
data = data[data.answers != '[]']
data = data.reset_index(drop=True)

print('data loaded')

my_dict = {}
i = 0

start = time.time()
columns = ['main_category', 'question', 'related_topics', 'sub_category', 'doctor_name', 'doctor_profession', 'short_answer', 'answer']
count = 0
data.apply(lambda s: transformRow(s), axis=1)

end = time.time()
print(end-start)

df = pd.DataFrame.from_dict(my_dict, "index")
df.to_csv('../../data/healthtap_full_processed.csv')