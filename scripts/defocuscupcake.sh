gpu=0
output_dir="deblurdata"
DATAPATH=...

scene=defocuscupcake
llffhold=6

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python train.py \
    -s ${DATAPATH}/${scene} \
    -m ../${output_dir}/${scene} \
    --init_dgt 0.0006 --iterations 30000 --ms_steps 10000 --min_opacity 0.1 \
    --eval -r 4 --port $(expr 6009 + $gpu) --kernel_size 0.1 \
    --llffhold ${llffhold}



