import torch
from super_gradients.training.dataloaders.dataloaders import (coco_detection_yolo_format_train, coco_detection_yolo_format_val)
from super_gradients.training import models
from super_gradients import Trainer, setup_device
from super_gradients.training import MultiGPUMode
from prettyformatter import pprint

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
CHECKPOINT_DIR = './checkpoints/'
setup_device(multi_gpu=MultiGPUMode.AUTO)
trainer = Trainer(experiment_name='fire_training', ckpt_root_dir=CHECKPOINT_DIR)

MODEL_ARCH = 'yolo_nas_m'

model = models.get(MODEL_ARCH, pretrained_weights="coco",num_classes=1).to(DEVICE)

BATCH_SIZE = 6
MAX_EPOCHS = 25

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': '/home/matthew/Personal/fire_ds',
        'images_dir': 'images/train',
        'labels_dir': 'labels/train',
        'classes': ["fire"]
    },
    dataloader_params={
        'batch_size': BATCH_SIZE,
        'num_workers': 4
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': '/home/matthew/Personal/fire_ds',
        'images_dir': 'images/val',
        'labels_dir': 'labels/val',
        'classes': ["fire"]
    },
    dataloader_params={
        'batch_size': BATCH_SIZE,
        'num_workers': 4
    }
)

print('Dataloader parameters:')
pprint(train_data.dataloader_params)
print('Dataset parameters')
pprint(train_data.dataset.dataset_params)

from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

train_params = {
    'silent_mode': False,
    "average_best_models":True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": MAX_EPOCHS,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        num_classes=1,
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=1,
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50'
}

trainer.train(
    model=model, 
    training_params=train_params, 
    train_loader=train_data, 
    valid_loader=val_data
)