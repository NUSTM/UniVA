import os
import torch
import time
from time import strftime
from utils.pinionGear import *
from utils.metrics import get_multi_labels_metrics

def train_detail(fabric, set_name, loader, hparams, 
                    model, model_optimizer=None, model_scheduler=None, 
                    multiLabels_criterion=None, vaAware_criterion=None, 
                    epoch=None):
    if set_name == 'train':
        model.train()
        is_testing = False
        num_batches = hparams.trg_n_train // hparams.trg_batch_size      
        total_size = 0
        start_time = time.time()
        model_optimizer.zero_grad()
    else:
        model.eval()
        is_testing = True if set_name == 'test' else False
        predictLabels_list = []
        truthLabels_list = []
    total_loss = 0
    
    for i_batch, batch in enumerate(loader): 
        batch_size = hparams.trg_batch_size 
        loss = 0

        if not hparams.vaAug:
            if hparams.text_plm_checkpoint != 'glove':
                text_inputs, text_att_mask, text_flag_mask, audio_inputs, audio_att_mask, vision_inputs, vision_att_mask, groundTruth_labels= batch
                groundTruth_text_va, groundTruth_audio_va, groundTruth_vision_va = None, None, None, None
            else:
                text_inputs, text_att_mask, audio_inputs, audio_att_mask, vision_inputs, vision_att_mask, groundTruth_labels= batch
                groundTruth_text_va, text_flag_mask, groundTruth_audio_va, groundTruth_vision_va = None, None, None, None, None
        else:
            if hparams.text_plm_checkpoint != 'glove':
                text_inputs, text_att_mask, text_flag_mask, audio_inputs, audio_att_mask, vision_inputs, vision_att_mask, groundTruth_labels, \
                                                        groundTruth_text_va, groundTruth_audio_va, groundTruth_vision_va= batch
            else:
                text_inputs, text_att_mask, audio_inputs, audio_att_mask, vision_inputs, vision_att_mask, groundTruth_labels, \
                                    groundTruth_text_va, groundTruth_audio_va, groundTruth_vision_va= batch
                text_flag_mask = None

        with torch.no_grad() if set_name != 'train' else dummy_context():
            predict_labels, multilabel_loss, va_SupCon_loss = model(text_inputs, text_att_mask, text_flag_mask, audio_inputs, audio_att_mask, 
                                                vision_inputs, vision_att_mask, is_testing, groundTruth_labels, multiLabels_criterion,
                                                groundTruth_text_va, groundTruth_audio_va, groundTruth_vision_va, vaAware_criterion)

        if set_name != 'test':
            if hparams.SupConLoss:
                multilabel_loss_update = (1-hparams.alpha) * multilabel_loss
                va_SupCon_loss_update = hparams.alpha * va_SupCon_loss
                loss = multilabel_loss_update + va_SupCon_loss_update
            else:
                loss = multilabel_loss
            loss = loss/ hparams.trg_accumulation_steps
        
        if set_name == 'train':
            fabric.backward(loss)
            if ((i_batch+1) % hparams.trg_accumulation_steps)==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.clip)
                model_optimizer.step() 
                model_scheduler.step()
                model_optimizer.zero_grad()

        total_loss += loss.item() * batch_size * hparams.trg_accumulation_steps if set_name != 'test' else 0
        
        if set_name == 'train':
            total_size += batch_size
            if i_batch % hparams.trg_log_interval == 0 and i_batch > 0:   
                avg_loss = total_loss / total_size
                elapsed_time = time.time() - start_time
                fabric.print('**TRG** | Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                    format(epoch, i_batch, num_batches, elapsed_time * 1000 / hparams.trg_log_interval, avg_loss))  
                total_loss, total_size = 0, 0
                start_time = time.time()
        else:
            predictLabels_list.append(predict_labels)
            truthLabels_list.append(groundTruth_labels)

    if set_name != 'train':
        avg_loss = total_loss / hparams.trg_n_valid if set_name == 'valid' else None
        predictLabels = torch.cat(predictLabels_list)
        truthLabels = torch.cat(truthLabels_list)
        return avg_loss, predictLabels, truthLabels


def trainer(fabric, trg_train_loader, trg_valid_loader, trg_test_loader,
            model, model_optimizer, model_scheduler, 
            multiLabels_criterion, vaAware_criterion, 
            hparams):
    
    best_model_time = 0
    best_val_score = float('-inf')

    for epoch in range(1, hparams.num_epochs+1):
        start = time.time()
        train_detail(fabric, 'train', trg_train_loader, hparams,  
                        model, model_optimizer, model_scheduler, 
                        multiLabels_criterion, vaAware_criterion, 
                        epoch)
        val_loss, predictLabels, truthLabels = train_detail(fabric, 'valid', trg_valid_loader, hparams, 
                        model, model_optimizer, model_scheduler, 
                        multiLabels_criterion, vaAware_criterion, 
                        epoch)
        end = time.time()
        duration = round((end-start)/60,4)
        fabric.print("*"*50)
        ml_acc, hanming_loss, micro_f1, macro_f1 = get_multi_labels_metrics(truthLabels, predictLabels)
        fabric.print(f'**TRG** | Epoch {epoch} | Time {duration}min | Val_Loss {val_loss:.4f} | ACC {ml_acc:.2f} | HL {hanming_loss:.3f} | miF1 {micro_f1:.2f} | maF1 {macro_f1:.2f}')
        
        fabric.print("-"*50)

        #save best model
        curr_epoch_score = micro_f1

        if curr_epoch_score > best_val_score:
            current_time = strftime("%m-%d-%H-%M-%S")
            save_checkpoint(fabric, hparams.choice_modality, model, hparams, current_time)
            best_val_score = curr_epoch_score
            best_model_time = current_time

    #testing
    checkpoint_model = load_checkpoint(fabric, hparams.choice_modality, hparams.save_model_path, best_model_time)
    model.load_state_dict(checkpoint_model, strict=False)
    _, predictLabels, truthLabels = train_detail(fabric, 'test', trg_test_loader, hparams, 
                                                        model)
    ml_acc, hanming_loss, micro_f1, macro_f1 = get_multi_labels_metrics(truthLabels, predictLabels)
    fabric.print(f'The results on the {hparams.choice_modality} modality:\n'
                f'Acc: {ml_acc:.2f},\nHL: {hanming_loss:.3f},\nmiF1: {micro_f1:.2f},\nmaF1: {macro_f1:.2f}\n')







