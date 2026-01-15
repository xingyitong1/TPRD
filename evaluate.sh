CUDA_VISIBLE_DEVICES=0 python projects/deformable_detr/train_net.py \
    --config-file "path/to/config.py" \
    --num-gpus 1 \
    --eval-only \
    train.init_checkpoint="path/to/checkpoint.pth" \