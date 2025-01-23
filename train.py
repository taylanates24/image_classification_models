import hydra
from omegaconf import DictConfig
from model import SceneClassifier
import torch
from data.dataset import CustomDataset
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
from get_optim import get_scheduler, LRSchedulerEnum
from classifier import Classifier
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from focal_loss import FocalLoss
from datetime import datetime
import os
from enum import Enum

class OptimizerEnum(Enum):
    ADAM = "adam"
    SGD = "sgd"
    ASGD = "asgd"

    @classmethod
    def from_value(cls, value):
        """
        Validate and return the enum member corresponding to the given value.

        Args:
            value (str): The optimizer name to validate.

        Returns:
            OptimizerEnum: The enum member if valid.

        Raises:
            ValueError: If the given value is not valid.
        """
        try:
            return cls(value)
        except ValueError:
            valid_values = ", ".join([member.value for member in cls])
            raise ValueError(f"Invalid optimizer: '{value}'. Valid options are: {valid_values}")

class LossEnum(Enum):
    CE = "CE"
    FOCAL = "focal"

    @classmethod
    def from_value(cls, value):
        """
        Validate and return the enum member corresponding to the given value.

        Args:
            value (str): The optimizer name to validate.

        Returns:
            OptimizerEnum: The enum member if valid.

        Raises:
            ValueError: If the given value is not valid.
        """
        try:
            return cls(value)
        except ValueError:
            valid_values = ", ".join([member.value for member in cls])
            raise ValueError(f"Invalid optimizer: '{value}'. Valid options are: {valid_values}")
        

@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg: DictConfig):
    # Extract configurations
    model_name = cfg.model.name
    pretrained = cfg.model.pretrained
    num_classes = cfg.model.num_classes

    optimizer_name = OptimizerEnum.from_value(cfg.optimizer).value
    experiment_path = cfg.experiment_path
    
    loss_name = LossEnum.from_value(cfg.loss).value
    w1 = cfg.w1
    w2 = 1 - w1
    gamma = cfg.gamma
    alpha = cfg.alpha
    weight_decay = cfg.weight_decay
    learning_rate = cfg.learning_rate
    val_frequency = cfg.val_frequency
    epochs = cfg.epochs

    # Dataset paths and parameters
    train_cfg = cfg.dataset.train
    val_cfg = cfg.dataset.val

    # Model setup
    model = SceneClassifier(pretrained=pretrained, model_name=model_name, num_classes=num_classes)

    # Dataset and DataLoader setup
    train_dataset = CustomDataset(image_dir=train_cfg.image_path, config_path=cfg, phase='train')
    val_dataset = CustomDataset(image_dir=val_cfg.image_path, config_path=cfg, phase='val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=train_cfg.shuffle,
        num_workers=train_cfg.num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_cfg.batch_size,
        shuffle=val_cfg.shuffle,
        num_workers=val_cfg.num_workers
    )

    # Loss function
    weights = torch.tensor([w1, w2])

    if loss_name == LossEnum.CE.value:
        loss = nn.CrossEntropyLoss(weight=weights)
    elif loss_name == LossEnum.FOCAL.value:
        loss = FocalLoss(alpha=alpha, gamma=gamma)

    # Logger and checkpoint paths
    current_dateTime = datetime.now()
    logger_path = f"model_{model_name}_bs_{train_cfg.batch_size}_img_size_{train_cfg.img_size}_opt_{optimizer_name}_loss_{loss_name}_date_{current_dateTime}"
    logger = TensorBoardLogger(experiment_path, name=logger_path)

    ckpt_path = os.path.join(experiment_path, logger_path)

    # Optimizer and scheduler
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if optimizer_name == OptimizerEnum.ADAM.value
        else torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        )
    
    scheduler_name = LRSchedulerEnum.from_value(cfg.lr_scheduler)
    scheduler = get_scheduler(scheduler_name, optimizer=optimizer, len_dataset=len(train_loader), num_epochs=epochs)
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
        img_size=train_cfg.img_size,
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

if __name__ == '__main__':
    main()
