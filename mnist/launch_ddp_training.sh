torchrun --nproc_per_node=1 --nnodes=2 --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT train.py
