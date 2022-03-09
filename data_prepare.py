# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""duee 1.0 dataset process"""
import os
import sys
import json
import random
from numpy.core.fromnumeric import mean
from numpy.core.overrides import set_module
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
# from torch.utils.data.dataset import T
from utils.utils import read_by_lines, write_by_lines, get_entities


def ke_convert_example_to_feature(example, tokenizer, label_vocab=None, max_seq_len=64, no_entity_label="O", ignore_label=-1, is_test=False):
    '''convert example to feature for keyword extraction
    '''
    tokens, labels = example
    tokenized_input = tokenizer(
        tokens,
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=max_seq_len)

    input_ids = tokenized_input['input_ids']
    token_type_ids = tokenized_input['token_type_ids']
    seq_len = len(tokens) + 2

    if is_test:
        return input_ids, token_type_ids, seq_len
    assert label_vocab is not None
    labels = labels[:(max_seq_len-2)]
    encoded_labels = [no_entity_label] + labels + [no_entity_label]
    encoded_labels = [label_vocab[x] for x in encoded_labels]

    # padding label 
    encoded_labels += [ignore_label] * (max_seq_len - len(encoded_labels))

    return input_ids, token_type_ids, seq_len, encoded_labels

def tm_convert_example_to_feature(example, tokenizer, max_seq_len=64, is_test=False):
    '''convert example to feature for text matching
    '''
    tokens_1, tokens_2, label = example
    tokenized_input = tokenizer(
        tokens_1,
        tokens_2,
        is_split_into_words=True,
        padding='max_length',
        truncation='only_first',
        max_length=max_seq_len)

    input_ids = tokenized_input['input_ids']
    token_type_ids = tokenized_input['token_type_ids']
    seq_len = len(tokens_1) + len(tokens_2) + 3
    if is_test:
        return input_ids, token_type_ids, seq_len

    return input_ids, token_type_ids, seq_len, label

class KeywordDataset(Dataset):
    """Keyword Extraction"""
    def __init__(self, data_path):
        
        self.all_words = []
        self.all_labels = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            # skip the head line
            next(fp)
            for line in fp.readlines():
                #print(line)
                #print(line.strip('\n').split('\t'))
                splitted = line.strip('\n').split('\t')
                words = splitted[0]
                labels = splitted[1] if len(splitted) > 1 else ''
                words = words.split('\002')
                labels = labels.split('\002')
                self.all_words.append(words)
                self.all_labels.append(labels)
                

    def __len__(self):
        return len(self.all_words)

    def __getitem__(self, index):
        return self.all_words[index], self.all_labels[index]

class TextMatchDataset(Dataset):
    """Keyword Extraction"""
    def __init__(self, data_path):
        
        self.all_words_1 = []
        self.all_words_2 = []
        self.all_labels = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            # skip the head line
            next(fp)
            for line in fp.readlines():
                #print(line)
                #print(line.strip('\n').split('\t'))
                splitted = line.strip('\n').split('\t')
                words_1, words_2 = splitted[0], splitted[1]
                label = splitted[2] if len(splitted) > 2 else '0'
                words_1 = words_1.split('\002')
                words_2 = words_2.split('\002')
                label = int(label.strip())
                self.all_words_1.append(words_1)
                self.all_words_2.append(words_2)
                self.all_labels.append(label)
                
    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, index):
        return self.all_words_1[index], self.all_words_2[index], self.all_labels[index]

def data_process_baidu_xueshu(path, keyword_type, is_predict=False, need_header=True):

    def label_data(data, start, length, _type):
        for i in range(start, start+length):
            suffix = 'B' if i == start else 'I'
            data[i] = f'{suffix}-{_type}'
        return data
    
    output = []
    not_matched_count = 0
    text_max_len = 0
    text_lens = []
    if need_header:
        output = ['text_a'] if is_predict else ['text_a\tlabel']
    with open(path, encoding='utf-8') as f:
        for line in f:
            d_json = json.loads(line.strip())
            keyword, paper_titles = list(d_json.keys())[0], list(d_json.values())[0]
            keyword = keyword.lower()
            for title in paper_titles:
                text_a = [
                    ' ' if t in ['\n', '\t'] else t 
                    for t in list(title.lower())
                ]
                if len(text_a) > 64: continue

                if is_predict:
                    output.append('\002'.join(text_a))
                else:
                    labels = ['O'] * len(text_a)
                    start = find_matched_start_index(text_a, keyword)
                    if start >= 0:
                        # text_max_len = len(text_a) if len(text_a) > text_max_len else text_max_len
                        text_lens.append(len(text_a))
                        # if len(text_a) > 100:
                        #     print(''.join(text_a))
                        labels = label_data(labels, start, len(keyword), keyword_type)
                        output.append('{}\t{}'.format('\002'.join(text_a), '\002'.join(labels)))
                    else:
                        not_matched_count += 1
    print(f'未匹配到关键词的样本数: {not_matched_count}，匹配到的有: {len(output)}')
    print(f'最大、最小和评价样本长度分别为: {max(text_lens)}、{min(text_lens)} 和 {mean(text_lens)}')

    return output

def data_process_journal(path, is_predict=False, need_header=True):

    def label_data(data, start, length, _type='keyword'):
        for i in range(start, start+length):
            suffix = 'B' if i == start else 'I'
            data[i] = f'{suffix}-{_type}'
        return data
    
    output = []
    text_lens = []
    if need_header:
        output = ['text_a'] if is_predict else ['text_a\tlabel']
    with open(path, encoding='utf-8') as f:
        for line in f:
            d_json = json.loads(line.strip())
            title, keyword_list = d_json['title'], d_json.get('keywords_matched', [])
            
            text_a = [
                ' ' if t in ['\n', '\t'] else t 
                for t in list(title.lower())
            ]
            if len(text_a) > 64 or len(text_a) < 5: continue
            # if len(text_a) < 5 or len(text_a) > 55: print(title)
            if is_predict:
                output.append('\002'.join(text_a))
            else:
                labels = ['O'] * len(text_a)
                for keyword in keyword_list:
                    start = find_matched_start_index(text_a, keyword)
                    if start >= 0:
                        # text_max_len = len(text_a) if len(text_a) > text_max_len else text_max_len
                        text_lens.append(len(text_a))
                        # if len(text_a) > 100:
                        #     print(''.join(text_a))
                        labels = label_data(labels, start, len(keyword))
                output.append('{}\t{}'.format('\002'.join(text_a), '\002'.join(labels)))
    if len(text_lens) > 0:
        print(f'最大、最小和评价样本长度分别为：{max(text_lens)}、{min(text_lens)} 和 {mean(text_lens)}')

    return output

def data_process_text_match(path, is_predict=False, need_header=True):

    output = []
    text_lens = []
    if need_header:
        output = ['text_a\ttext_b'] if is_predict else ['text_a\ttext_b\tlabel']
    with open(path, encoding='utf-8') as f:
        for line in f:
            d_json = json.loads(line.strip())
            title, keyword_list = d_json['title'], d_json.get('keywords', [])
            
            text_a = [
                ' ' if t in ['\n', '\t'] else t 
                for t in list(title.lower())
            ]
            if not is_predict and (len(text_a) > 64 or len(text_a)) < 4: continue
            for keyword in keyword_list:

                text_b = [
                    ' ' if t in ['\n', '\t'] else t
                    for t in list(keyword.lower())
                ]
                if not is_predict and (len(text_b) > 10 or len(text_b)) < 3: continue
                text_lens.append(len(text_a)+len(text_b))
                if len(text_a) + len(text_b) > 100:
                    print('here.')
                # if len(text_a) < 5 or len(text_a) > 55: print(title)
                if is_predict:
                    output.append('{}\t{}'.format('\002'.join(text_a), '\002'.join(text_b)))
                else:   
                    output.append('{}\t{}\t{}'.format('\002'.join(text_a), '\002'.join(text_b), 1))
    if len(text_lens) > 0:
        print(f'最大、最小和评价样本长度分别为: {max(text_lens)}、{min(text_lens)} 和 {mean(text_lens)}')

    return output

def min_edit_distance_ratio(text_a, text_b):
    n = len(text_a)
    m = len(text_b)
    assert m + n > 0

    return min_edit_distance(text_a, text_b) / max(m, n)

def min_edit_distance(text_a, text_b):
    '''计算两字符串(列表)之间的最小编辑距离
    source: https://leetcode-cn.com/problems/edit-distance/solution/bian-ji-ju-chi-by-leetcode-solution/
    '''
    n = len(text_a)
    m = len(text_b)
    # 有一个字符串为空串
    if n * m == 0:
        return n + m
    # DP 数组
    D = [ [0] * (m + 1) for _ in range(n + 1)]
    # 边界状态初始化
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    # 计算所有 DP 值
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            left = D[i - 1][j] + 1
            down = D[i][j - 1] + 1
            left_down = D[i - 1][j - 1] 
            if text_a[i - 1] != text_b[j - 1]:
                left_down += 1
            D[i][j] = min(left, down, left_down)
    
    return D[n][m]

def find_matched_start_index(char_list, word):
    '''在 char_list 里找到 word 的连续字符，并返回位置的开头索引，若找不到，即返回 -1'''
    for i, char_ in enumerate(char_list):
        if char_ == word[0] and i <= len(char_list)-len(word):
            # min_distance = min_edit_distance(char_list[i:i+len(word)], word)
            # if min_distance < 2: # 可能存在一个字的差别
            #     if min_distance > 0:
            #         print(f'编辑距离大于 0')
            #     return i
            # else:
            #     print(f'{char_list[i:i+len(word)]} 与 {word} 编辑距离太长： {min_distance}')
            if all(char_list[i+j] == word[j] for j in range(1, len(word))):
                # print(f"{''.join(char_list[i:i+len(word)])} 与 {word}")
                return i
    # print('未找到')

    return -1

def data_augmentation_for_text_match(train_data, negative_ratio=3):
    '''数据增强: 为文本匹配的训练集增加负样本, 增强后正样本与负样本的比例大致为 1: negative_ratio
    '''
    auged_train_data = []
    for i, sample in enumerate(train_data):
        auged_train_data.append(sample)
        splitted = sample.strip('\n').split('\t')
        words_1, words_2 = splitted[0], splitted[1]

        random_indices = random.sample(range(len(train_data)), negative_ratio)
        for ind in random_indices:
            if ind == i: continue
            splitted = train_data[ind].strip('\n').split('\t')
            r_words_1, r_words_2 = splitted[0], splitted[1]
            auged_train_data.append('{}\t{}\t{}'.format(words_1.strip(), r_words_2.strip(), 0))
    
    print(f'增强前数量:{len(train_data)}, 增强后: {len(auged_train_data)}, 增长倍数: {len(auged_train_data) / len(train_data)}')
    return auged_train_data


def split_train_test(all_data, data_save_path, ratio_train=0.8):
    '''Split data into train, dev and test sets according to the ratio_train.'''
    header = all_data[:1]
    train_data, test_data = train_test_split(all_data[1:], train_size=ratio_train, random_state=24, shuffle=True)
    write_by_lines(os.path.join(data_save_path, 'train.tsv'), header + train_data)
    write_by_lines(os.path.join(data_save_path, 'test.tsv'), header + test_data)
    print(f'train_size: {len(train_data)}, test_size: {len(test_data)}')

if __name__ == "__main__":


    # text = '在 char_list 里找到 word 的连续字符，并返回位置的开头索引，若找不到，即返回 -1'
    # print(find_matched_start_index(list(text), 'word'))


    print("\n=================Annotating==============")
    
    ''' for baidu xueshu
    
    root = os.path.dirname(__file__)
    data_root = os.path.join(root, 'data')
    raw_data_root = os.path.join(data_root, 'matched_paper_from_baidu_xueshu')
    keyword_types = ['method', 'scope', 'topic']
    outputs = []
    for i, k_type in enumerate(keyword_types):

        file_path = os.path.join(raw_data_root, 'matched_{}_paper_titles_all.jl'.format(k_type))
        k_output = data_process_baidu_xueshu(file_path, k_type, is_predict=False, need_header=i==0)
        outputs.extend(k_output)
    # 划分数据集
    split_train_test(outputs, data_root, ratio_train=0.8)
    '''
    
    '''for journals
    data_root = 'data/journals'
    data_file = os.path.join(data_root, 'preprocessed_data.jl')
    output = data_process_journal(data_file, need_header=True)
    split_train_test(output, data_root, ratio_train=0.8)
    print(f'all_data: {len(output)}')
    '''
    
    '''for conference
    data_root = 'data/conference'
    data_file = os.path.join(data_root, 'report_titles.jl')
    output = data_process_journal(data_file, is_predict=True, need_header=True)
    write_by_lines(os.path.join(data_root, 'predict.tsv'), output)
    print(f'all_data: {len(output)}')
    '''

    # for text matching
    data_root = 'data/journals'
    data_file = os.path.join(data_root, 'preprocessed_data.jl')
    # data_file = 'data/text_match/predict_topics.jl'
    all_data = data_process_text_match(data_file, is_predict=False, need_header=True)
    # write_by_lines('data/text_match/predict.tsv', all_data)
    header = all_data[:1]
    train_data, test_data = train_test_split(all_data[1:], train_size=0.8, random_state=24, shuffle=True)
    
    output_folder = 'data/text_match'
    for n_p_ratio in [3, 5, 8, 10, 15, 20]:
        auged_train_data = data_augmentation_for_text_match(train_data, negative_ratio=n_p_ratio)
        write_by_lines(os.path.join(output_folder, f'train_{n_p_ratio}.tsv'), header + auged_train_data)
    write_by_lines(os.path.join(output_folder, 'test.tsv'), header + test_data)
    print(f'原始数据集长度: {len(all_data)-1}, 测试集长度: {len(test_data)}')
