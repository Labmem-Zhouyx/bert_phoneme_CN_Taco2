import json
import codecs

with codecs.open('./uni_class2idx.json', 'r', 'utf-8') as usernames:
    cedict = json.load(usernames)

all_pinyins = []
for key in cedict.keys():
    all_pinyins.append(key)

file = []
with open('mel-bznsyp_character_pinyin_data_train.txt') as f:
    file.extend(f.readlines())
with open('mel-bznsyp_character_pinyin_data_test.txt') as f:
    file.extend(f.readlines())
with open('mel-bznsyp_character_pinyin_data_val.txt') as f:
    file.extend(f.readlines())

for sen in file:
    pinyin_seq = sen.split('|')[2].strip()
    pinyins = pinyin_seq.split(' ')
    all_pinyins.extend(pinyins)

class2idx = {}
pinyins_dict = list(set(all_pinyins))

for i in range(len(pinyins_dict)):
    class2idx[pinyins_dict[i]] = i

with codecs.open('./bznsyp_pinyin2idx.json', 'w', 'utf-8') as usernames:
    json.dump(class2idx, usernames, ensure_ascii=False)