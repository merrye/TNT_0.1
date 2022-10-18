# bash script to record the training config

# for test the program on the complete dataset
# python train_net.py -d ../TNT-Trajectory-Prediction-main/dataset/interm_data -o run/net/ -a -b 64 -c --lr 0.001 -luf 10 -ldr 0.1

# for test the program on the small dataset
# python train_net.py -d dataset/interm_data_small -o run/net/ -a -b 128 -c --lr 0.0010 -luf 10 -ldr 0.1

# for multi-gpu training
# nproc_per_node: set the number of gpu
python -m torch.distributed.launch --nproc_per_node=1 train_net.py -d dataset/interm_data_small -o run/net/ -a -b 128 -c -m --lr 0.0012 -luf 10 -ldr 0.3

# when you need to choose the training GPU
# CUDA_VISIBLE_DEVICES=1,0 python -m torch.distributed.launch --nproc_per_node=1 train_net.py -d ../TNT-Trajectory-Prediction-main/dataset/interm_data -o run/net/ -a -b 128 -c -m --lr 0.0012 -luf 10 -ldr 0.3