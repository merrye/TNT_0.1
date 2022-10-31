# bash script to record the training config

# for test the program on the complete dataset
#python train_tnt.py -d dataset/interm_data -o run/tnt/ -a -b 64 -c --lr 0.0010 -luf 10 -ldr 0.1 -e 100 -w 40

# for test the program on the small dataset
# python train_tnt.py -d dataset/interm_data -o run/tnt/ -a -b 128 -c --lr 0.0010 -luf 10 -ldr 0.1

# for multi-gpu training
# nproc_per_node: set the number of gpu
 python -m torch.distributed.launch --nproc_per_node=2 train_tnt.py -d dataset/interm_data_small -o run/tnt/ -a -b 200 -c -m --lr 0.0012 -luf 10 -ldr 0.3 -w 40

# when you need to choose the training GPU
# CUDA_VISIBLE_DEVICES=1,0 python -m torch.distributed.launch --nproc_per_node=2 train_tnt.py -d dataset/interm_data_small -o run/tnt/ -a -b 128 -c -m --lr 0.0012 -luf 10 -ldr 0.3
