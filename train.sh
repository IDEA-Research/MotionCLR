accelerate --mixed_precision "no" launch --config_file 1gpu.yaml --gpu_ids 0 -m scripts.train --name release --model-ema --dataset_name t2m --dropout 0.1 --lr 1e-4 --no_eff --self_attention --edit_mode
