import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
import torch
from .ExtraTextDetail import Data_Text
from .pinionGear import normalized_labels_va


def load_modality_Feat_uttMask(args, split_type, modality, modality_type, vaAug):
    if modality_type !='text':
        file_path = os.path.join(args.data_prepro_path, 'final_feat', modality, f'{args.DATASET}_{split_type}_{modality_type}_utt_VA.pkl')
    else:
        file_path = os.path.join(args.data_prepro_path, 'final_feat', modality, f'{args.DATASET}_{split_type}_{modality_type}_glove300_VA.pkl')
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
    if modality_type !='text':
        feature = torch.tensor(dataset[split_type][modality_type].astype(np.float32)).cpu().detach()
        utt_mask = torch.tensor(dataset[split_type][f'{modality_type}_utt_mask'].astype(np.float32)).cpu().detach()
    else:
        feature = torch.tensor(dataset[modality_type].astype(np.float32)).cpu().detach()
        utt_mask = torch.tensor(dataset[f'{modality_type}_utt_mask'].astype(np.float32)).cpu().detach()      
    
    if vaAug:
        if modality_type !='text':
            va_values_tmp = torch.tensor(np.array(dataset[split_type][f'{modality_type}_VA_values']).astype(np.float32)).cpu().detach()
        else:
            va_values_tmp = torch.tensor(np.array(dataset[f'{modality_type}_VA_values']).astype(np.float32)).cpu().detach()           
        va_values = normalized_labels_va(va_values_tmp)
    else:
        va_values = None

    if modality_type != 'text':
        multi_labels = torch.tensor(dataset[split_type][f'labels_class'].astype(np.float32)).cpu().detach()
        num_multi_labels = dataset['num_labels']
    else:
        multi_labels, num_multi_labels = None, None

    file.close()
    return feature, utt_mask, multi_labels, num_multi_labels, va_values


class loading_multimodal_data(Dataset):
    def __init__(self, text_features, num_utt, num_labels, split_type, text_plm_name, args):
        super(loading_multimodal_data, self).__init__()

        self.vaAug = args.vaAug
        self.choice_modality = args.choice_modality
        self.text_plm_name = text_plm_name
        '''textual modality'''
        if text_plm_name != 'glove':
            self.features = text_features
            self.text_input_ids = torch.tensor([f.input_ids for f in self.features], dtype=torch.long) # (num_utt, 512)
            self.text_input_mask = torch.tensor([f.input_mask for f in self.features], dtype=torch.long) # (num_utt, 512)
            self.text_flag_mask = torch.tensor([f.flag_mask for f in self.features], dtype=torch.long)  # (num_utt, 512)
            # self.multi_labels = torch.tensor([f.multi_labels for f in self.features], dtype=torch.float32) # (num_utt, num_labels)
            if self.vaAug:
                text_va_values_tmp = torch.tensor([f.va_values for f in self.features], dtype=torch.float32)  # (num_utt, 2)
                self.text_va_values = normalized_labels_va(text_va_values_tmp)
                self.belong_diaID = [f.belong_dia_id for f in self.features]  # (num_utt, )
        else:
            self.text_feature, self.text_utt_mask, _, _, text_va_values_tmp = load_modality_Feat_uttMask(args, split_type, 'T', 'text', self.vaAug)
            if self.vaAug:
                self.text_va_values = normalized_labels_va(text_va_values_tmp)
                self.belong_diaID = None

        self.audio_feature, self.audio_utt_mask, multi_labels, num_multi_labels, self.audio_va_values = load_modality_Feat_uttMask(args, split_type, 'A', 'audio', self.vaAug)
        self.vision_feature, self.vision_utt_mask, multi_labels, num_multi_labels, self.vision_va_values = load_modality_Feat_uttMask(args, split_type, 'V', 'vision', self.vaAug)

        self.num_utt = self.audio_feature.shape[0]
        self.num_labels = num_multi_labels
        self.multi_labels = multi_labels


    def __len__(self):
        return self.num_utt      #返回utterance的总个数

    def get_text_max_utt_len(self):
        return self.text_feature.shape[1]  
    
    def get_text_featExtr_dim(self):
        return self.text_feature.shape[-1]    

    def get_audio_max_utt_len(self):
        return self.audio_feature.shape[1]  
    
    def get_audio_featExtr_dim(self):
        return self.audio_feature.shape[-1]    

    def get_vision_max_utt_len(self):
        return self.vision_feature.shape[1]  
    
    def get_vision_featExtr_dim(self):
        return self.vision_feature.shape[-1]  

    def get_data_num_labels(self):
        return self.num_labels

    def __getitem__(self, index):    
        
        if self.text_plm_name != 'glove':
            batch_text_inputs = self.text_input_ids[index]
            batch_text_att_mask = self.text_input_mask[index]
            batch_text_flag_mask = self.text_flag_mask[index]
        else:
            batch_text_feat = self.text_feature[index]
            batch_text_att_mask = self.text_utt_mask[index]

        batch_class_labels = self.multi_labels[index]

        if self.vaAug:
            text_va_values = self.text_va_values[index]
            audio_va_values = self.audio_va_values[index]
            vision_va_values = self.vision_va_values[index]
        
        batch_audio_inputs = self.audio_feature[index]  
        batch_audio_utt_mask = self.audio_utt_mask[index]

        batch_vision_inputs = self.vision_feature[index]  
        batch_vision_utt_mask = self.vision_utt_mask[index]

        if not self.vaAug:
            if self.text_plm_name != 'glove':
                return batch_text_inputs, batch_text_att_mask, batch_text_flag_mask, batch_audio_inputs, batch_audio_utt_mask, \
                        batch_vision_inputs, batch_vision_utt_mask, batch_class_labels
            else:
                return batch_text_feat, batch_text_att_mask, batch_audio_inputs, batch_audio_utt_mask, \
                        batch_vision_inputs, batch_vision_utt_mask, batch_class_labels
        else:
            if self.text_plm_name != 'glove':
                return batch_text_inputs, batch_text_att_mask, batch_text_flag_mask, batch_audio_inputs, batch_audio_utt_mask, \
                        batch_vision_inputs, batch_vision_utt_mask, batch_class_labels, text_va_values, audio_va_values, vision_va_values
            else:
                return batch_text_feat, batch_text_att_mask, batch_audio_inputs, batch_audio_utt_mask, \
                        batch_vision_inputs, batch_vision_utt_mask, batch_class_labels, text_va_values, audio_va_values, vision_va_values

def get_multimodal_data_detail(args,split,vaAug, text_plm_name):
    if text_plm_name != 'glove':
        data_text = Data_Text(split, vaAug, text_plm_name, args)
        data_text_features, num_utt, num_labels = data_text.preprocess_data()
        return loading_multimodal_data(data_text_features, num_utt, num_labels, split, text_plm_name, args)
    else:
        return loading_multimodal_data(None, None, None, split, text_plm_name, args)

def get_multimodal_data(args, split=None, vaAug=False):
    if args.text_plm_checkpoint == 'chinese-roberta-wwm-ext':
        text_plm_name = 'bert'
    elif args.text_plm_checkpoint == 'roberta-base':
        text_plm_name = 'roberta'
    else:
        text_plm_name = 'glove'
    
    data_save_path = os.path.join(args.data_prepro_path, 'final_feat', args.choice_modality, 'dt_folder', text_plm_name)
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    data_path = os.path.join(data_save_path, f'{args.DATASET}_{args.choice_modality}_{split}_{args.text_plm_checkpoint}_{args.KLDivLoss}_{vaAug}.dt')
    print(f'loading {args.DATASET} multimodal_'+args.choice_modality+'_'+split+'...')
    
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = get_multimodal_data_detail(args,split,vaAug, text_plm_name)
        torch.save(data, data_path, pickle_protocol=4)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path, map_location=torch.device('cpu'))
    return data

