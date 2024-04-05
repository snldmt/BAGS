gpu=0
output_dir="deblurdata"
DATAPATH=...

scene=defocusseal
llffhold=6

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python train.py \
    -s ${DATAPATH}/${scene} \
    -m ../${output_dir}/${scene} \
    --init_dgt 0.0002 --iterations 10000 --ms_steps 7000 --min_opacity 0.1 \
    --depth_loss_alpha 0.01 --rgbtv_loss_alpha 0.01 --use_depth_loss --use_rgbtv_loss \
    --eval -r 4 --port $(expr 6009 + $gpu) --kernel_size 0.1 \
    --llffhold ${llffhold}



