gpu=0
output_dir="deblurdata"
DATAPATH=...

scene=defocuscoral
llffhold=10

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python train.py \
    -s ${DATAPATH}/${scene} \
    -m ../${output_dir}/${scene} \
    --init_dgt 0.0006 --iterations 22000 --ms_steps 18000 --min_opacity 0.1 \
    --use_another_mlp \
    --eval -r 4 --port $(expr 6009 + $gpu) --kernel_size 0.1 \
    --llffhold ${llffhold}



