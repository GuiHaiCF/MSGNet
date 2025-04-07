# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi

# if [ ! -d "./logs/ETTh1" ]; then
#     mkdir ./logs/ETTh1
# fi
export CUDA_VISIBLE_DEVICES=0,1

seq_len=12
label_len=6
# model_name=MSGNet
model_name=Autoformer

pred_len=12
python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ECG.csv \
    --model_id ECG'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --freq s \
    --target '0' \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 140 \
    --dec_in 140 \
    --c_out 140 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 64 \
    --top_k 3 \
    --conv_channel 32 \
    --skip_channel 32 \
    --dropout 0.1 \
    --batch_size 32 \
    --itr 1  #>logs/ETTh1/$model_name'_'ETTh1_$seq_len'_'$pred_len.log
