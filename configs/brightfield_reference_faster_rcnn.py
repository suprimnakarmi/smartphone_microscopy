import os.path

_base_ = (
    "../mmdetection/configs/faster_rcnn/faster_rcnn_x101_32x8d_fpn_mstrain_3x_coco.py"
)

# sample type
sample_type = "brightfield_reference"

# dataset settings
dataset_type = "CocoDataset"
data_root = "/mnt/Enterprise/safal/AI_assisted_microscopy_system/cysts_dataset_all/brightfield_reference"
classes = ("Crypto", "Giardia")
# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="RepeatDataset",
        times=3,
        dataset=dict(
            type=dataset_type,
            classes=classes,
            ann_file=os.path.join(
                data_root, "brightfield_reference_coco_annos_train.json"
            ),
            img_prefix=os.path.join(data_root, "train"),
        ),
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=os.path.join(data_root, "brightfield_reference_coco_annos_val.json"),
        img_prefix=os.path.join(data_root, "train"),
    ),
)

# change the number of classes in roi head to match the dataset
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2),
    ),
)

checkpoint_config = dict(interval=1, max_keep_ckpts=3)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type="TensorboardLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="mmdetection_cysts",
                name=f"{sample_type}_faster_rcnn_x101_32x8d_fpn_mstrain_3x_coco",
            ),
        ),
    ],
)

work_dir = f"/mnt/Enterprise/safal/AI_assisted_microscopy_system/outputs/{sample_type}/faster_rcnn_x101_32x8d_fpn_mstrain_3x_coco"
