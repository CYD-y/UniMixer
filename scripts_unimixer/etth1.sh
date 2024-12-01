export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96 \
    --model UniMixer \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 16 \
    --n_heads 8 \
    --d_ff 16 \
    --dropout 0.1 \
    --itr 1 \
    --lradj constant \
    --train_epoch 50 \
    --patience 3 \
    --e_layers 1 \
    --k 32

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_192 \
    --model UniMixer \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 16 \
    --n_heads 8 \
    --d_ff 128 \
    --dropout 0.1 \
    --itr 1 \
    --lradj constant \
    --train_epoch 50 \
    --patience 2 \
    --e_layers 1 \
    --k 32

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_336 \
    --model UniMixer \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --batch_size 128 \
    --learning_rate 1e-3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 16 \
    --n_heads 8 \
    --d_ff 32 \
    --dropout 0.1 \
    --itr 1 \
    --lradj constant \
    --train_epoch 50 \
    --patience 2 \
    --e_layers 1 \
    --k 32

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_720 \
    --model UniMixer \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 16 \
    --n_heads 8 \
    --d_ff 64 \
    --dropout 0.1 \
    --itr 1 \
    --lradj type3 \
    --train_epoch 50 \
    --patience 6 \
    --e_layers 1 \
    --k 32