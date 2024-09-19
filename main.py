# -*- encoding: utf-8 -*-

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true" 
import argparse
import lightning as L
import transformers
import torch
from torch.utils.data import DataLoader
from utils.dataset import get_multimodal_data
from models import *
from train import trainer
from utils.vaAware_supcon_loss import VA_aware_SupConLoss

'''NUSTM server'''
# DATA_PATH = '/root/data1/000_dataset'
DATA_PATH = '/media/devin/data/'
# PRETRA_MODEL_PATH = '/root/data1/000_dataset/public_pretrained_models'
PRETRA_MODEL_PATH = '/media/devin/data/public_pretrained_model/'

if not os.path.exists(PRETRA_MODEL_PATH):
    raise FileNotFoundError("Please modify the path where the pre-trained models are stored...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Please modify the path where the dataset is stored...")

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='A Unimodal VA-driven Multimodal Multi-label Emotion Recognition')
#---------------------------------------------------------------------------------------------------------------------------------------------#
#data
parser.add_argument('--DATASET', type=str, default="MOSEI", choices=["MOSEI", "M3ED" ])
parser.add_argument('--choice_modality', type=str, default='T+A+V', help='[T+A+V, T+A, T, A, V]')
parser.add_argument('--precision', type=str, default="32", choices=["32", "16-mixed"], help='precision of the model')

parser.add_argument('--KLDivLoss', type=bool, default=True, help='whether to use KL-divergence loss or BCE loss')
parser.add_argument('--thres_kl', type=float, default=0.18, help='threshold to transform 0-1 for multi-label distribution')

parser.add_argument('--SupConLoss', action='store_true', default=False, help='whether to use Supervised Contrastive Learning Loss')
parser.add_argument('--temp', type=float, default=0.6, help='temperature for SupCon loss function')

parser.add_argument('--vaAug', action='store_true', default=False, help='whether to use multi-dim VA labels to augment.')
parser.add_argument('--thres_dist', type=float, default=0.1, help='threshold for nearby samples to determine the positive pairs')

parser.add_argument('--labelSimilar_regulari', action='store_true', default=False, help='whether to use Label Relation Prior')
parser.add_argument('--alpha', type=float, default=0.4, help='weight for SupCon loss and multi-label classification loss')
#----------------------------------------------------------------
#loading pretrained model
parser.add_argument('--save_model_path', default= os.path.join(PROJECT_PATH, 'save'))
parser.add_argument('--public_plm_path', type=str, default=PRETRA_MODEL_PATH, help='the path of public text pretrained model')
parser.add_argument('--text_plm_checkpoint', type=str, default="glove", choices=["roberta-base", "roberta-large", "chinese-roberta-wwm-ext", "glove"], 
                            help='the pretrained language model checkpoint')
#---------------------------------------------------------------------------------------------------------------------------------------------#
#tuning
parser.add_argument('--num_epochs', type=int, default=15,  
                    help='number of epochs')
parser.add_argument('--trg_lr', type=float, default=7e-5,   
                    help='initial learning rate (default: )')
parser.add_argument('--trg_batch_size', type=int, default=32, 
                    help='num of utterance of dataset') 
parser.add_argument('--trg_accumulation_steps',type=int, default=1,  
                    help='gradient accumulation for trg task')
#----------------------------------------------------------------
#public model args
parser.add_argument('--hidden_size', type=int, default=768, help='input size for transformer/conformer')
parser.add_argument('--fc_dropout', type=float, default=0.1, help='dropout rate of the fully connected layer') 
parser.add_argument('--final_dropout',type=float, default=0.1, help='the drop rate of the last layer')
#----------------------------------------------------------------
parser.add_argument('--text_num_transformer_layers', type=int, default=4, help='transformer encoder layers(transformer-L, default: 12)')
parser.add_argument('--audio_num_transformer_layers', type=int, default=4, help='transformer encoder layers(transformer-L, default: 12)')
parser.add_argument('--vision_num_transformer_layers', type=int, default=3, help='transformer encoder layers(transformer-L, default: 12)')
parser.add_argument('--num_attention_heads', type=int, default=12, help='number of heads for the transformer network, 12')  
parser.add_argument('--intermediate_size', type=int, default=3072, help='embedding intermediate layer size, 4*hidden_size, 3072')
parser.add_argument('--hidden_act', type=str, default='gelu', help='non-linear activation function')
parser.add_argument('--hidden_dropout_prob',type=float, default=0.1, help='hidden layer dropout')
parser.add_argument('--attention_probs_dropout_prob',type=float, default=0.1,help='attention dropout')
parser.add_argument('--layer_norm_eps', type=float, default=1e-12, help='1e-12')  
parser.add_argument('--initializer_range',type=int, default=0.02)
#---------------------------------------------------------------------------------------------------------------------------------------------#
'''logistics'''
parser.add_argument('--n_gpus', type=int, default=4, help="the number of gpus")
parser.add_argument('--warm_up', type=float, default=0.1, help='dynamic adjust learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')  
parser.add_argument('--clip', type=float, default=1, help='gradient clip value (default: 1)')  
parser.add_argument('--trg_log_interval', type=int, default=400, help='frequency of result logging')  
parser.add_argument("--seed", type=int, default=1111, help="random seed (default: 1111)")

args = parser.parse_args()

args.text_plm_path = os.path.join(args.public_plm_path, args.text_plm_checkpoint)
args.data_path = os.path.join(DATA_PATH, args.DATASET)
args.data_prepro_path = os.path.join(DATA_PATH, args.DATASET, 'preprocess_data')

if __name__ == '__main__':
    #####################################################################################################################################################################
    '''loading fabric'''
    L.seed_everything(args.seed)
    fabric = L.Fabric(accelerator="gpu", devices=args.n_gpus, precision=args.precision)
    fabric.launch()

    if args.text_plm_checkpoint == 'glove':
        if torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 3090 Ti' or args.n_gpus != 1:
            fabric.print("Warning: If you use the glove model, you need to use 1 3090 Ti GPU, otherwise you may not get the results in the paper, it is recommended to re-tune...")
    else:
        if torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 3090' or args.n_gpus != 4:
            fabric.print('Warning: If you use the RoBERTa model, you need to use 4 3090 GPUs, otherwise you may not get the results in the paper, it is recommended to re-tune...')
        
    fabric.print('*'*100)
    fabric.print(f'Modeling for the {args.DATASET} dataset is about to begin...')

    if args.SupConLoss:
        fabric.print('*'*50)
        fabric.print(f'Adding Contrastive Learning Loss...')
    if args.vaAug:
        fabric.print('*'*50)
        fabric.print(f'Adding multi-dimensional VA emotion information to guide positive and negative sample pairs...')
    if args.labelSimilar_regulari:
        fabric.print('*'*50)
        fabric.print(f'Adding label relation prior...')
    fabric.print('*'*50)
    fabric.print(f'Using {args.text_plm_checkpoint}')
    fabric.print('*'*50)

    #####################################################################################################################################################################
    '''loading dataset'''
    fabric.print('Preparing to perform a MMER task using text, audio, and visual data...')
    print('*'*50)
    trg_train_data = get_multimodal_data(args, 'train', args.vaAug)  
    trg_valid_data = get_multimodal_data(args, 'valid', args.vaAug)
    trg_test_data = get_multimodal_data(args, 'test', args.vaAug)

    trg_train_loader = DataLoader(trg_train_data, batch_size=args.trg_batch_size, shuffle=True, num_workers=1)
    trg_valid_loader = DataLoader(trg_valid_data, batch_size=args.trg_batch_size, shuffle=False, num_workers=1)
    trg_test_loader = DataLoader(trg_test_data, batch_size=args.trg_batch_size, shuffle=False, num_workers=1)
    args.trg_n_train, args.trg_n_valid, args.trg_n_test = len(trg_train_data), len(trg_valid_data), len(trg_test_data)
    args.num_multi_labels = trg_train_data.get_data_num_labels()  

    if args.text_plm_checkpoint == 'glove':
        args.text_feat_dim = trg_train_data.get_text_featExtr_dim()
        args.get_text_utt_max_lens = max(trg_train_data.get_text_max_utt_len(),trg_valid_data.get_text_max_utt_len(),trg_test_data.get_text_max_utt_len())  

    args.audio_feat_dim = trg_train_data.get_audio_featExtr_dim()
    args.get_audio_utt_max_lens = max(trg_train_data.get_audio_max_utt_len(),trg_valid_data.get_audio_max_utt_len(),trg_test_data.get_audio_max_utt_len())  

    args.vision_feat_dim = trg_train_data.get_vision_featExtr_dim()
    args.get_vision_utt_max_lens = max(trg_train_data.get_vision_max_utt_len(),trg_valid_data.get_vision_max_utt_len(),trg_test_data.get_vision_max_utt_len())  

    trg_train_loader = fabric.setup_dataloaders(trg_train_loader)
    trg_valid_loader = fabric.setup_dataloaders(trg_valid_loader)
    trg_test_loader  = fabric.setup_dataloaders(trg_test_loader)

    #####################################################################################################################################################################
    '''load model and optimizer'''
    model = multimodal_t_a_v_model(args)
    
    model_optimizer = transformers.AdamW(model.parameters(), lr=args.trg_lr, weight_decay=args.weight_decay)  
    model, model_optimizer = fabric.setup(model, model_optimizer)  

    multiLabels_criterion = torch.nn.KLDivLoss(reduction='mean')
    vaAware_criterion = VA_aware_SupConLoss(temperature=args.temp) if args.SupConLoss else None

    total_training_steps = args.num_epochs * len(trg_train_loader) / args.trg_accumulation_steps
    model_scheduler = transformers.get_linear_schedule_with_warmup(optimizer = model_optimizer,
                                                num_warmup_steps = int(total_training_steps * args.warm_up),
                                                num_training_steps = total_training_steps)
    
    #####################################################################################################################################################################
    '''train and test'''
    trainer(fabric, trg_train_loader, trg_valid_loader, trg_test_loader,
            model, model_optimizer, model_scheduler, 
            multiLabels_criterion, vaAware_criterion, 
            args)



