python -u main.py \
--gpus 1 \
--model resnet20_1w1a \
--results_dir ./result \
--data_path /data_path \
--dataset cifar10 \
--epochs 600 \
--lr 0.1 \
-b 128 \
-bt 128 \
--lr_type cos \
--weight_decay 1e-4 \
--tau_min 0.85 \
--tau_max 0.99 \
--seed 0 \