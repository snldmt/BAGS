gpu=0
output_dir="mip360_mixres"
DATAPATH=...

scene=treehill

export KERLR=0.001
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python train.py \
    -s ${DATAPATH} \
    -m ../${output_dir}/${scene} \
    --init_dgt 0.0006 --iterations 35000 --ms_steps 12000 --min_opacity 0.005 --init_opacity 0.1 \
    --kernel_size_ss 5 --use_rgbtv_loss --use_depth_loss --depth_loss_alpha 0.01 \
    --eval -r 4 --port $(expr 6009 + $gpu) --kernel_size 0.1 --not_use_rgbd




