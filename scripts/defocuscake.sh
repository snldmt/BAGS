gpu=0
output_dir="deblurdata"
DATAPATH=...

scene=defocuscake
llffhold=9

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python train.py \
    -s ${DATAPATH}/${scene} \
    -m ../${output_dir}/${scene} \
    --init_dgt 0.0008 --iterations 58000 --ms_steps 12000 --min_opacity 0.1 \
    --use_depth_loss --use_rgbtv_loss --use_another_mlp \
    --eval -r 4 --port $(expr 6009 + $gpu) --kernel_size 0.1 \
    --llffhold ${llffhold}



