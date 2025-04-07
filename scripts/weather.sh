# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi

# if [ ! -d "./logs/weather" ]; then
#     mkdir ./logs/weather
# fi
export CUDA_VISIBLE_DEVICES=0,1

seq_len=96
label_len=48
model_name=MSGNet
# model_name=DLinear

pred_len=96
python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id weather'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --freq t \
    --target 'OT' \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 64 \
    --top_k 3 \
    --conv_channel 32 \
    --skip_channel 32 \
    --dropout 0.1 \
    --batch_size 32 \
    --itr 1 #>logs/electricity/$model_name'_'electricity_$seq_len'_'$pred_len.log