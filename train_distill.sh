CUDA_VISIBLE_DEVICES=0,1 python projects/deformable_detr/train_net.py \
    --config-file projects/deformable_detr/cfg_distill/DETRDistill_TPRD/deformable_detr_r50_distll_r18_50ep.py \
    --num-gpus 2 \
    train.output_dir="your output directory" \
    dataloader.train.total_batch_size=16
