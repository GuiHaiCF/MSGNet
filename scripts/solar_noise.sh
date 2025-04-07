#export CUDA_VISIBLE_DEVICES=0,1

# 只使用cuda:0 (物理卡号0)
export CUDA_VISIBLE_DEVICES=0

# # 或只使用cuda:1 (物理卡号1)
# export CUDA_VISIBLE_DEVICES=1

seq_len=12
label_len=6
model_name=Autoformer


pred_len=12
python -u noise_test.py \
    --is_training 0 \
    --root_path ./dataset/ \
    --data_path solar.csv \
    --model_id solar'_'$seq_len'_'$pred_len \
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
    --enc_in 216 \
    --dec_in 216 \
    --c_out 216 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 64 \
    --top_k 3 \
    --conv_channel 32 \
    --skip_channel 32 \
    --dropout 0.1 \
    --batch_size 32 \
    --itr 1 #>logs/electricity/$model_name'_'electricity_$seq_len'_'$pred_len.log
