import argparse
import yaml
from model import SceneClassifier
import torch
from data.augment import Augmentations
from data.dataset import FPDataset, VideoTypeDataset
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cfg', type=str, default='train.yaml', help='training config file')
    
    args = parser.parse_args()
    
    opt = args.train_cfg
    with open(opt, 'r') as config:
        opt = yaml.safe_load(config)    
    
    model_name = opt['model_name']
    optimizer_name = opt['optimizer']
    experiment_path = opt['experiment_path']
    num_classes = opt['num_classes']
    loss_name = opt['loss']

    w1 = opt['w1']
    #w2 = opt['w2']
    w2 = 1 - w1
    model = SceneClassifier(pretrained=False, model_name=model_name, num_classes=num_classes)

    pred_detections_pth = opt['pred_detections_pth']
    bbox_infer_dataset = opt['bbox_infer_dataset']
    annotations_pth = opt['annotations_pth']

    train_image_dir = opt['dataset']['train']['image_path']
    train_img_size = opt['dataset']['train']['img_size']
    train_augmentations = [Augmentations(opt['dataset']['augmentations'], img_size=train_img_size)]
    train_batch_size = opt['dataset']['train']['batch_size']

    val_image_dir = opt['dataset']['val']['image_path']
    val_img_size = opt['dataset']['val']['img_size']
    val_batch_size = opt['dataset']['val']['batch_size']
    val_augmentations = [Augmentations(opt['dataset']['val_augmentations'], img_size=val_img_size)]


    gamma = opt['gamma']
    alpha = opt['alpha']
    weight_decay = opt['weight_decay']
    learning_rate = opt['learning_rate']

    train_dataset = VideoTypeDataset(image_path=train_image_dir,
                              augment=True,
                              augmentations=train_augmentations,
                                  image_size=train_img_size)
    
    val_dataset = VideoTypeDataset(image_path=val_image_dir,
                            augmentations=val_augmentations,
                                  image_size=val_img_size)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=opt['dataset']['train']['shuffle'],
        num_workers=opt['dataset']['train']['num_workers'] 
        )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=val_batch_size,
        num_workers=opt['dataset']['val']['num_workers'] 
    )
    
    current_dateTime = datetime.now()
    weights = torch.tensor([w1, w2])

    if loss_name == 'CE':
        loss = nn.CrossEntropyLoss(weight=weights) 
        
    elif loss_name == 'focal':
        loss = FocalLoss(alpha=alpha, gamma=gamma)
    #loss = FocalLoss(alpha=alpha, gamma=gamma)
    #loss = nn.CrossEntropyLoss(weight=weights)

    logger_path = f'model_{model_name}_bs_{train_batch_size}_img_size_{train_img_size}_opt_{optimizer_name}_loss_{loss_name}_date_{current_dateTime}' 
    
    logger = TensorBoardLogger(experiment_path, name=logger_path)


    #optimizer = get_optimizer(opt, model, wd=weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) if optimizer_name =='adam' else \
                        torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    scheduler = get_scheduler(opt, optimizer, len(train_loader))
    
    early_stopping_callback = EarlyStopping(monitor='average precision', patience=100)

    checkpoint_callback = ModelCheckpoint(
        dirpath="/workspaces/scene_classification/",
        filename="best-checkpoint2",
        save_top_k=1,
        verbose=True,
        monitor="average precision",
        mode="max")

    ckpt_path = os.path.join(experiment_path, logger_path)

    classifier = Classifier(model=model, scheduler=scheduler, optimizer=optimizer, loss=loss, ckpt_path=ckpt_path, num_classes=num_classes, 
                            img_size=train_img_size, pred_detections_pth=pred_detections_pth,
                            annotations_pth=annotations_pth, bbox_infer_dataset=bbox_infer_dataset, 
                            model_name=model_name)

    trainer = pl.Trainer(gpus=1,
                         logger=logger,
                         callbacks=[checkpoint_callback],
                         #resume_from_checkpoint='/workspaces/FaceShapeClassifier/checkpoints/best-checkpoint.ckpt', 
                         check_val_every_n_epoch=opt['val_frequency'], 
                         max_epochs=opt['epochs'])

    trainer.fit(model=classifier, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader)
    
    #classifier.test_tensorrt()
    print('asd')
    # trainer.test(model=classifier, 
    #             dataloaders=val_loader, 
    #             ckpt_path='/workspaces/scene_classification/best-checkpoint2.ckpt')
    
        # def test(
        # self,
        # model: Optional["pl.LightningModule"] = None,
        # dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        # ckpt_path: Optional[str] = None,
        # verbose: bool = True,
        # datamodule: Optional[LightningDataModule] = None,