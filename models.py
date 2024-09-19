import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel
from modules.Transformer import TransformerEncoder, AdditiveAttention
from utils.pinionGear import getBinaryTensor


class multimodal_t_a_v_model(nn.Module):
    def __init__(self, hparams) -> None:
        super(multimodal_t_a_v_model, self).__init__()
        self.DATASET = hparams.DATASET

        self.text_plm_checkpoint = hparams.text_plm_checkpoint
        
        self.KLDivLoss = hparams.KLDivLoss
        self.thres_kl = hparams.thres_kl
        
        self.SupConLoss = hparams.SupConLoss
        self.thres_dist = hparams.thres_dist

        self.labelSimilar_regulari = hparams.labelSimilar_regulari
        
        '''loading textual modality'''
        if self.text_plm_checkpoint != 'glove':
            self.text_plm = AutoModel.from_pretrained(hparams.text_plm_path) 
            self.text_plm.pooler.dense.bias.requires_grad=False
            self.text_plm.pooler.dense.weight.requires_grad=False
            self.text_linear_plm = nn.Sequential(
                                    nn.Linear(self.text_plm.config.hidden_size, hparams.hidden_size),
                                    nn.Dropout(hparams.fc_dropout)
                                    )
        else:
            self.text_linear_glove = nn.Sequential(
                                    nn.Linear(300, hparams.hidden_size),
                                    nn.Dropout(hparams.fc_dropout)
                                    )

            self.text_utt_level_transformer = TransformerEncoder(hparams, hparams.text_num_transformer_layers, hparams.get_text_utt_max_lens, hparams.hidden_size)   
        self.text_attention_mapping = AdditiveAttention(hparams.hidden_size, hparams.hidden_size)
        
        '''loading acoustic modality'''
        self.audio_linear = nn.Sequential(
                                    nn.Linear(hparams.audio_feat_dim, hparams.hidden_size),
                                    nn.Dropout(hparams.fc_dropout)
                                    )
        self.audio_utt_level_transformer = TransformerEncoder(hparams, hparams.audio_num_transformer_layers, hparams.get_audio_utt_max_lens, hparams.hidden_size)   
        self.audio_attention_mapping = AdditiveAttention(hparams.hidden_size, hparams.hidden_size)
        
        '''loading visual modality'''
        self.vision_linear = nn.Sequential(
                                    nn.Linear(hparams.vision_feat_dim, hparams.hidden_size),
                                    nn.Dropout(hparams.fc_dropout)
                                    )
        self.vision_utt_level_transformer = TransformerEncoder(hparams, hparams.vision_num_transformer_layers, hparams.get_vision_utt_max_lens, hparams.hidden_size)   
        self.vision_attention_mapping = AdditiveAttention(hparams.hidden_size, hparams.hidden_size)


        self.fc = nn.Sequential(
                            nn.Linear(hparams.hidden_size + hparams.hidden_size + hparams.hidden_size, hparams.num_multi_labels),
                            nn.Dropout(hparams.final_dropout)
                            )

        '''loading labels similarity matrix M'''
        if self.labelSimilar_regulari:
            utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils')
            label_similarity_matrix_path = os.path.join(utils_dir, f'{self.DATASET}_label_similarity_matrix.npy')
            self.label_similar_matrix = nn.Parameter(torch.tensor(np.load(label_similarity_matrix_path), dtype=torch.float32,requires_grad=True).cuda())

        '''loading classifier'''
        self.classifier = nn.Sigmoid() if not self.KLDivLoss else nn.LogSoftmax()
        
    def forward(self, text_inputs=None, text_att_mask=None, text_flag_mask=None,
                audio_inputs=None, audio_att_mask=None,
                vision_inputs=None, vision_att_mask=None,
                is_testing=None,
                groundTruth_labels=None, multi_label_criterion=None,
                groundTruth_text_va=None,
                groundTruth_audio_va=None, groundTruth_vision_va=None,
                vaAware_criterion=None):

        batch_size = text_inputs.shape[0]

        if self.text_plm_checkpoint != 'glove':
            dia_outputs = self.text_plm(text_inputs, text_att_mask)[0]
            batch_utt_feats = torch.zeros_like(dia_outputs).cuda()   # (batch_size, max_utt_len, feat_dim)
            batch_utt_masks = torch.zeros_like(text_att_mask).cuda() # (batch_size, max_utt_len)
            for i in range(batch_size):
                valid_utt_indices = text_flag_mask[i].bool()  # (max_utt_len, ) value为True or False
                curr_utt_feat = dia_outputs[i][valid_utt_indices]  
                curr_utt_mask = text_att_mask[i][valid_utt_indices] 
                batch_utt_feats[i, :curr_utt_feat.size(0)] = curr_utt_feat
                batch_utt_masks[i, :curr_utt_mask.size(0)] = curr_utt_mask
            text_utt_embed = self.text_linear_plm(batch_utt_feats) #(batch_size, max_seq_len, hidden_size)    
            text_utt_feat, _ = self.text_attention_mapping(text_utt_embed, batch_utt_masks)
        else:
            text_utt_embed = self.text_linear_glove(text_inputs) #(batch_size, max_seq_len, hidden_size)    
            text_extended_att_mask = text_att_mask.unsqueeze(1).unsqueeze(2)
            text_extended_att_mask = text_extended_att_mask.to(dtype=next(self.parameters()).dtype)
            text_extended_att_mask = (1.0 - text_extended_att_mask) * -10000.0  
            batch_utt_feats = self.text_utt_level_transformer(text_utt_embed, text_extended_att_mask)
            text_utt_feat, _ = self.text_attention_mapping(batch_utt_feats, text_att_mask)

        audio_extended_att_mask = audio_att_mask.unsqueeze(1).unsqueeze(2)
        audio_extended_att_mask = audio_extended_att_mask.to(dtype=next(self.parameters()).dtype)
        audio_extended_att_mask = (1.0 - audio_extended_att_mask) * -10000.0  
        audio_emb_linear = self.audio_linear(audio_inputs)
        audio_utt_trans = self.audio_utt_level_transformer(audio_emb_linear, audio_extended_att_mask)  #(batch_size, utt_max_lens, self.hidden_size)
        audio_utt_feat, _ = self.audio_attention_mapping(audio_utt_trans, audio_att_mask) #(batch_size, hidden_size)

        vision_extended_att_mask = vision_att_mask.unsqueeze(1).unsqueeze(2)
        vision_extended_att_mask = vision_extended_att_mask.to(dtype=next(self.parameters()).dtype)
        vision_extended_att_mask = (1.0 - vision_extended_att_mask) * -10000.0  
        vision_emb_linear = self.vision_linear(vision_inputs)
        vision_utt_trans = self.vision_utt_level_transformer(vision_emb_linear, vision_extended_att_mask)  #(batch_size, utt_max_lens, self.hidden_size)
        vision_utt_feat, _ = self.vision_attention_mapping(vision_utt_trans, vision_att_mask) #(batch_size, hidden_size)
        
        multimodal_feat = torch.cat((text_utt_feat, audio_utt_feat, vision_utt_feat), dim=-1) 

        multimodal_fc = self.fc(multimodal_feat)   
        predict_scores = self.classifier(multimodal_fc) #(batch_size, num_multi_labels)

        if self.labelSimilar_regulari:
            label_diff = predict_scores.unsqueeze(2) - predict_scores.unsqueeze(1)
            label_norm = torch.norm(label_diff, p=2, dim=0)
            regular_term = torch.sum(label_norm * self.label_similar_matrix) / batch_size
            # print(f'regular_term: {regular_term}')
        else:
            regular_term = 0
        
        predict_labels = getBinaryTensor(torch.exp(predict_scores), self.thres_kl) 
        groundTruth_scores = F.softmax(groundTruth_labels, dim=-1)
        
        multilabel_label_similarity_loss, va_SupCon_loss = None, None
        if is_testing == False:
            multilabel_loss = multi_label_criterion(predict_scores, groundTruth_scores) * 1000
            multilabel_label_similarity_loss = multilabel_loss + regular_term

            if self.SupConLoss:
                batch_text_feat_final = torch.stack((text_utt_feat,text_utt_feat), dim=1)
                batch_audio_feat_final = torch.stack((audio_utt_feat,audio_utt_feat), dim=1)
                batch_vision_feat_final = torch.stack((vision_utt_feat,vision_utt_feat), dim=1)
                text_va_SupCon_loss = vaAware_criterion(batch_text_feat_final, None, groundTruth_text_va, self.thres_dist)
                audio_va_SupCon_loss = vaAware_criterion(batch_audio_feat_final, None, groundTruth_audio_va, self.thres_dist)
                vision_va_SupCon_loss = vaAware_criterion(batch_vision_feat_final, None, groundTruth_vision_va, self.thres_dist)
                if self.DATASET == "MOSEI":
                    va_SupCon_loss = (3 * text_va_SupCon_loss + audio_va_SupCon_loss + vision_va_SupCon_loss) / 3 
                else:
                    va_SupCon_loss = (text_va_SupCon_loss + audio_va_SupCon_loss + vision_va_SupCon_loss) / 3 
        return predict_labels, multilabel_label_similarity_loss, va_SupCon_loss
