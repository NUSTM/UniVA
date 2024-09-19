#coding:utf-8
import os.path as osp
import pandas as pd
import json
from collections import defaultdict
from transformers import AutoTokenizer
MAX_SEQ_LEN = 512


def calcu_dia_tokens_len(curr_SetStru, curr_SetData, tokenizer):
    total_dia_lens = defaultdict(list)
    total_dia_tokens = defaultdict(list)
    for dia_id in list(curr_SetStru.keys()):
        curr_set_dia_list = curr_SetStru[dia_id]
        curr_set_dia_len = len(curr_set_dia_list)
        total_dia_lens[dia_id].append(curr_set_dia_len)
        curr_dia_tokens_len = 0
        for utt_id in curr_set_dia_list:
            utt_text = curr_SetData[utt_id]['text']
            utt_tokens = tokenizer.tokenize(utt_text)
            total_dia_tokens[dia_id].append(utt_tokens)
            curr_dia_tokens_len += len(utt_tokens)
        total_dia_lens[dia_id].append(curr_dia_tokens_len)
    return total_dia_lens, total_dia_tokens

def load_data_from_csv(file_path, set_name, curr_SetStru):
    data = pd.read_csv(file_path)

    set_data = data[data['mode'] == set_name]

    # Create the required dictionary
    set_data_dict = {
        f"{row['video_id']}_{row['clip_id']}": {
            "truely_index": row["truely_sort"],
            "belong_dia_lens": len(curr_SetStru[row['video_id']]),
            "text": row["text"],
            "happy": row["happy"],
            "sad": row["sad"],
            "anger": row["anger"],
            "surprise": row["surprise"],
            "disgust": row["disgust"],
            "fear": row["fear"],
        }
        for index, row in set_data.iterrows()
    }
    return set_data_dict

def make_labels(curr_SetData, data_name):

    labels = defaultdict(list)
    if data_name == 'MOSEI':
        emotion_mapping ={'happy': 0,
                    'sad': 1,
                    'anger': 2,
                    'surprise': 3,
                    'disgust': 4,
                    'fear': 5}
    else:
        emotion_mapping ={'Happy': 0, 'Surprise': 1, 'Sad': 2, 'Disgust': 3, 'Anger': 4, 'Fear': 5, 'Neutral': 6}
    num_labels = len(emotion_mapping)  
    total_utt_num = len(curr_SetData)

    for utt_id in list(curr_SetData.keys()):
        curr_utt = curr_SetData[utt_id]
        curr_utt_emotion = list(curr_utt['Emotion'].split(',')) if data_name != 'MOSEI' else None
        for label in emotion_mapping.keys():
            if data_name == 'MOSEI':
                labels[utt_id].append(1) if curr_utt[label] != 0 else labels[utt_id].append(0)
            else:
                if label in curr_utt_emotion:
                    labels[utt_id].append(1)
                else:
                    labels[utt_id].append(0)
                
    return labels, num_labels, total_utt_num


def _truncate_seq_pair(tokens, max_length):
    while True:
        tokens_len = []
        for i,utt in enumerate(tokens):
            tokens_len.append((i, len(utt)))
        # print("*"*10, tokens_len)
        sumlen = sum([i[1] for i in tokens_len])
        # print(sumlen)

        if sumlen <= max_length:   
            break
        else:
            index = sorted(tokens_len, key=lambda x:x[1], reverse=True)[0][0]
            # print(index)
            tokens[index].pop()
            # print(tokens)
    return tokens


from dataclasses import dataclass
@dataclass
class InputFeatures:
    input_ids: list
    input_mask: list 
    flag_mask: list
    multi_labels: list

@dataclass
class InputFeatures_va:
    input_ids: list
    input_mask: list 
    flag_mask: list
    multi_labels: list
    va_values: list
    belong_dia_id: list

class Data_Text():
    def __init__(self, split, vadAug, text_plm_name, args):
        
        self.set_name = split
        self.vadAug = vadAug
        self.text_plm_name = text_plm_name   
        self.DATASET = args.DATASET
        self.data_path = args.data_path
        self.data_prepro_path = args.data_prepro_path
    
        self.text_plm_path = args.text_plm_path

    def preprocess_data(self):
        tokenizer = AutoTokenizer.from_pretrained(self.text_plm_path)
        curr_SetStru = json.load(open(osp.join(self.data_prepro_path,'videos_structure.json'), 'r'))[self.set_name]
        if self.DATASET == 'MOSEI':
            curr_SetData = load_data_from_csv(osp.join(self.data_path,'final_multi_labels.csv'), self.set_name, curr_SetStru)
        else:
            curr_SetData = json.load(open(osp.join(self.data_prepro_path, 'utt_annot.json'), 'r'))[self.set_name]
        curr_set_labels, num_labels, curr_set_num_utt = make_labels(curr_SetData, self.DATASET)
        if self.vadAug:
            curr_set_VA = json.load(open(osp.join(self.data_prepro_path, 'raw_utt_text_textVA.json'), 'r'))[self.set_name]

        features = []
        flag_mask = []  #记住当前utt的每个词的位置
        tokens = []
        
        #统计每个dia的长度
        total_dia_lens, total_dia_tokens = calcu_dia_tokens_len(curr_SetStru, curr_SetData, tokenizer)

        for utt_id in list(curr_SetData.keys()):
            utt = curr_SetData[utt_id] #加载当前utt的信息
            belong_dia_id = utt_id.rsplit("_", 1)[0]   #所属的dia的名称
            
            if self.DATASET == 'MOSEI':
                belong_dia_lens = utt['belong_dia_lens']   #所属的dia的长度
                truely_index = utt['truely_index']
            else:
                belong_dia_lens = len(curr_SetStru[belong_dia_id])
                truely_index = curr_SetStru[belong_dia_id].index(utt_id)
            
            if total_dia_lens[belong_dia_id][1] < 460:
                max_tokens_len = MAX_SEQ_LEN
            else:
                if self.text_plm_name == 'roberta':
                    max_tokens_len = MAX_SEQ_LEN - total_dia_lens[belong_dia_id][0] * 2  #</s>utt_text</s></s>utt_text</s>
                elif self.text_plm_name == 'bert':
                    max_tokens_len = MAX_SEQ_LEN - total_dia_lens[belong_dia_id][0] - 1 # [CLS]utt_text[SEP]utt_text[SEP]
            tokens_temp = _truncate_seq_pair(total_dia_tokens[belong_dia_id], max_tokens_len)  

            '''记录当前utt的每个词的位置'''
            for ind, tokens_utt in enumerate(tokens_temp):
                if ind == 0:
                    tokens = ["<s>"] + tokens_utt + ["</s>"] if self.text_plm_name == 'roberta' else ["[CLS]"] + tokens_utt + ["[SEP]"]
                else:
                    tokens += ["</s>"] + tokens_utt + ["</s>"] if self.text_plm_name == 'roberta' else tokens_utt + ["[SEP]"]
            if truely_index == 0: #the first utt in the dialogue
                flag_mask = [0] + [1] * len(tokens_temp[truely_index]) + [0] * (len(tokens) - len(tokens_temp[truely_index]) - 1)
            else:
                if truely_index == belong_dia_lens -1 : ##the last utt in the dialogue
                    flag_mask = [0] * (len(tokens) - len(tokens_temp[truely_index]) - 1) + [1] * len(tokens_temp[truely_index]) + [0]
                else:
                    pre_utt_len = 0
                    for i in range(truely_index):
                        pre_utt_len += len(tokens_temp[i])
                    pre_utt_len += truely_index * 2 + 1 if self.text_plm_name == 'roberta' else truely_index + 1 #stats the length of the previous utterances
                    flag_mask = [0] * pre_utt_len + [1] * len(tokens_temp[truely_index]) + [0] * (len(tokens) - len(tokens_temp[truely_index]) - pre_utt_len)

            multi_labels = curr_set_labels[utt_id]

            if self.vadAug:
                if self.DATASET == 'MOSEI':
                    va_values = [eval(i) for i in curr_set_VA[utt_id][1:]]
                else:
                    va_values = [eval(i) for i in curr_set_VA[utt_id][2:]]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)  #当前dia长度的位置上置1

            # Zero-pad up to the sequence length. 
            padding = [0] * (MAX_SEQ_LEN - len(input_ids))  
            input_ids += padding
            input_mask += padding
            flag_mask += padding

            if self.vadAug:
                features.append(
                InputFeatures_va(input_ids=input_ids,
                            input_mask=input_mask,
                            flag_mask=flag_mask,
                            multi_labels=multi_labels,
                            va_values=va_values,
                            belong_dia_id=belong_dia_id))
            else:
                features.append(
                InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                flag_mask=flag_mask,
                                multi_labels=multi_labels))
        return features, curr_set_num_utt, num_labels

