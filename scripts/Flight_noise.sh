if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Flight" ]; then
    mkdir ./logs/Flight
fi

export CUDA_VISIBLE_DEVICES=0,1

seq_len=12
label_len=6
# model_name=MSGNet
model_name=Autoformer

for pred_len in 12
do
  python -u noise_test.py \
      --is_training 0 \
      --root_path ./dataset/ \
      --data_path Flight.csv \
      --model_id Flight'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --freq h \
      --target 'UUEE' \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1 \
      --d_model 512 \
      --d_ff 32 \
      --top_k 3 \
      --conv_channel 32 \
      --skip_channel 32 \
      --dropout 0.1 \
      --batch_size 32  #>logs/Flight/$model_name'_'Flight_$seq_len'_'$pred_len.log

done

