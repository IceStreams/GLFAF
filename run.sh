#!/usr/bin/env bash

python main.py --gpu 0 --data_name "SA" --scale 400 --max_epoch 1000 --hidden_size 64
#python main.py --gpu 0 --data_name "LongKou" --scale 200 --max_epoch 1000 --hidden_size 64
#python main.py --gpu 0 --data_name "PU" --scale 300 --max_epoch 1000 --hidden_size 64