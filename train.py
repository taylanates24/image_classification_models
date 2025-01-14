import argparse
import yaml
from model import SceneClassifier
import torch
from data.augment import Augmentations
from data.dataset import CustomDataset
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
from get_optim import get_optimizer, get_scheduler
from classifier import Classifier
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from focal_loss import FocalLoss
from datetime import datetime
import os

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cfg', type=str, default='train.yaml', help='training config file')
    args = parser.parse_args()

    # Load configuration
    cfg_file = args.train_cfg
    with open(cfg_file, 'r') as config:
        opt = yaml.safe_load(config)

    # Extract configurations
    model_name = opt['model_name']
    optimizer_name = opt['optimizer']
    experiment_path = opt['experiment_path']
    num_classes = opt['num_classes']
    loss_name = opt['loss']
    w1 = opt['w1']
    w2 = 1 - w1
    gamma = opt['gamma']
    alpha = opt['alpha']
    weight_decay = opt['weight_decay']
    learning_rate = opt['learning_rate']
    val_frequency = opt['val_frequency']
    epochs = opt['epochs']

    # Dataset paths and parameters
    train_image_dir = opt['dataset']['train']['image_path']
    train_img_size = opt['dataset']['train']['img_size']
    train_batch_size = opt['dataset']['train']['batch_size']
    val_image_dir = opt['dataset']['val']['image_path']
    val_img_size = opt['dataset']['val']['img_size']
    val_batch_size = opt['dataset']['val']['batch_size']

    # Model setup
    model = SceneClassifier(pretrained=True, model_name=model_name, num_classes=num_classes)

    # Dataset and DataLoader setup
    train_dataset = CustomDataset(image_dir=train_image_dir, config_path=cfg_file, phase='train')
    val_dataset = CustomDataset(image_dir=val_image_dir, config_path=cfg_file, phase='val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=opt['dataset']['train']['shuffle'],
        num_workers=opt['dataset']['train']['num_workers']
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=opt['dataset']['val']['shuffle'],
        num_workers=opt['dataset']['val']['num_workers']
    )

    # Loss function
    weights = torch.tensor([w1, w2])
    if loss_name == 'CE':
        loss = nn.CrossEntropyLoss(weight=weights)
    elif loss_name == 'focal':
        loss = FocalLoss(alpha=alpha, gamma=gamma)

    # Logger and checkpoint paths
    current_dateTime = datetime.now()
    logger_path = f"model_{model_name}_bs_{train_batch_size}_img_size_{train_img_size}_opt_{optimizer_name}_loss_{loss_name}_date_{current_dateTime}"
    logger = TensorBoardLogger(experiment_path, name=logger_path)

    ckpt_path = os.path.join(experiment_path, logger_path)

    # Optimizer and scheduler
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if optimizer_name == 'adam'
        else torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    )
    scheduler = get_scheduler(opt, optimizer, len(train_loader))

    # Callbacks
    early_stopping_callback = EarlyStopping(monitor='average precision', patience=100)
    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_path,
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="average precision",
        mode="max"
    )

    # Classifier setup
    classifier = Classifier(
        model=model,
        scheduler=scheduler,
        optimizer=optimizer,
        loss=loss,
        ckpt_path=ckpt_path,
        num_classes=num_classes,
        img_size=train_img_size,
        pred_detections_pth=opt['pred_detections_pth'],
        annotations_pth=opt['annotations_pth'],
        bbox_infer_dataset=opt['bbox_infer_dataset'],
        model_name=model_name
    )

    # Trainer setup
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        check_val_every_n_epoch=val_frequency,
        max_epochs=epochs
    )

    # Training
    trainer.fit(model=classifier, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print('Training completed.')
