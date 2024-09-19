# 1 x NVIDIA 3090Ti
curr_time=$(date +"%Y-%m-%d_%H-%M-%S")
echo 'current_time:' ${curr_time}

python -u main.py --DATASET MOSEI --choice_modality T+A+V --text_plm_checkpoint glove --n_gpus 1 --precision 32 --SupConLoss --vaAug --num_epochs 15 --trg_lr 7e-5 --trg_batch_size 32 --text_num_transformer_layers 7 --audio_num_transformer_layers 3 --vision_num_transformer_layers 5 --temp 0.4 --alpha 0.2