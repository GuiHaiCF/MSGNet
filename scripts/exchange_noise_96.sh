export CUDA_VISIBLE_DEVICES=0

seq_len=96
label_len=48
# model_name=MSGNet
model_name=CrossGNN

pred_len=96
python -u noise_test.py \
    --is_training 0 \
    --root_path ./dataset/ \
    --data_path exchange_rate.csv \
    --model_id exchange'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --freq h \
    --target '1' \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --d_model 64 \
    --d_ff 128 \
    --top_k 3 \
    --dropout 0.2 \
    --conv_channel 16 \
    --skip_channel 32 \
    --batch_size 32 \
    --itr 1 
