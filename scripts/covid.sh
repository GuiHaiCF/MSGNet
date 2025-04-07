# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi

# if [ ! -d "./logs/covid" ]; then
#     mkdir ./logs/covid
# fi

export CUDA_VISIBLE_DEVICES=0,1

seq_len=12
label_len=4
# model_name=MSGNet
model_name=DLinear

pred_len=12
python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path covid.csv \
    --model_id covid'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --target 'Alameda' \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 60 \
    --dec_in 60 \
    --c_out 60 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 64 \
    --top_k 3 \
    --conv_channel 32 \
    --skip_channel 32 \
    --dropout 0.3 \
    --batch_size 2 \
    --itr 1 #>logs/ETTm2/$model_name'_'ETTm2_$seq_len'_'$pred_len.log


