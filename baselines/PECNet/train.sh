python train.py\
 --width 300\
 --height 200\
 --gpus $1,\
 --lr 1e-4\
 --num_workers 4\
 --batch_size 32\
 --max_epochs 50\
 --data_dir "../../data"\
 --encoder "ResNet-18"\
 --save_dir "PECNet"\
 --use_ref_obj\
 --patience 10\
 --num_path_nodes 20\
 --unrolled\
 --neighbor_dist_thresh 0.1\
 --enc_layout_interm_size 512\
 --enc_layout_latent_size 256 128\
 --enc_layout_output_size 128\
 --enc_command_latent_size 256 128\
 --enc_command_output_size 128\
 --enc_dest_latent_size 8 16\
 --enc_latent_size 128 50\
 --non_local_theta_size 256 128 64\
 --non_local_phi_size 256 128 64\
 --non_local_g_size 256 128 64\
 --dec_latent_size 512 256 512\
 --predictor_latent_size 1024 512 256\
 --non_local_dim 128\
 --zdim 16\
 --fdim 16\
 --sigma 0.13\
 --kld_reg 0.001\
 --adl_reg 1.0\
 --non_local_pools 3\
 --input_type "layout"

#  --pecnet_config "/export/home2/NoCsBack/hci/dusan/Projects/Talk2Car_Path/baselines/trajectory/pecnet/config/config.yaml"\
# enc_latent_size - used to be [8, 50], and now is [128, 50]


# --width 300\
# --height 200\
# --gpus $1,\
# --lr 1e-4\
# --num_workers 4\
# --batch_size 32\
# --max_epochs 50\
# --data_dir "/cw/liir_code/NoCsBack/thierry/PathProjection/data_root"\
# --encoder "ResNet-18"\
# --save_dir "/home2/NoCsBack/hci/dusan/TrajectoryTesting/PECNet"\
# --use_ref_obj\
# --patience 10\
# --num_path_nodes 20\
# --unrolled\
# --neighbor_dist_thresh 0.1\
# --enc_layout_interm_size 512\
# --enc_layout_latent_size 256 128\
# --enc_layout_output_size 128\
# --enc_command_latent_size 256 128\
# --enc_command_output_size 128\
# --enc_dest_latent_size 8 16\
# --enc_latent_size 128 50\
# --non_local_theta_size 256 128 64\
# --non_local_phi_size 256 128 64\
# --non_local_g_size 256 128 64\
# --dec_latent_size 512 256 512\
# --predictor_latent_size 1024 512 256\
# --non_local_dim 128\
# --zdim 16\
# --fdim 16\
# --sigma 1.3\
# --kld_reg 0.01\
# --adl_reg 1.0\
# --non_local_pools 3\
# --input_type "locs"