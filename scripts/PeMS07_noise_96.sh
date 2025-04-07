
export CUDA_VISIBLE_DEVICES=0

seq_len=96
label_len=48
# model_name=MSGNet
model_name=Autoformer

pred_len=96
python -u noise_test.py \
    --is_training 0 \
    --root_path ./dataset/ \
    --data_path PeMS07.csv \
    --model_id PeMS07'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --freq t \
    --target '1' \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 228 \
    --dec_in 228 \
    --c_out 228 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 64 \
    --top_k 3 \
    --conv_channel 32 \
    --skip_channel 32 \
    --dropout 0.1 \
    --batch_size 32 \
    --itr 1 

