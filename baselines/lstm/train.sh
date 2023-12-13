#!/usr/bin/env bash
python train.py\
 --width 300\
 --height 200\
 --gpus $1,\
 --lr 1e-4\
 --num_workers 4\
 --batch_size 16\
 --max_epochs 50\
 --data_dir "/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root"\
 --encoder "ResNet-18"\
 --dataset "Talk2Car_Detector"\
 --save_dir "/export/home2/NoCsBack/hci/dusan/TrajectoryResults/LSTMBaselineFixedPathLength"\
 --use_ref_obj\
 --patience 10\
 --num_path_nodes 20\
 --embedding_dim 64\
 --hidden_dim 512
