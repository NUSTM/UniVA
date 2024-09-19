# 4 x NVIDIA 3090
curr_time=$(date +"%Y-%m-%d_%H-%M-%S")
echo 'current_time:' ${curr_time}

python -u main.py --DATASET M3ED --choice_modality T+A+V --text_plm_checkpoint chinese-roberta-wwm-ext --n_gpus 4 --precision 32 --SupConLoss --vaAug --labelSimilar_regulari --num_epochs 10 --trg_lr 5e-5 --trg_batch_size 22 --audio_num_transformer_layers 5 --vision_num_transformer_layers 3 --temp 0.4 --alpha 0.2