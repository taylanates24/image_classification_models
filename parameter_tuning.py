import torch
import pytorch_lightning as pl
import torch.nn as nn
from model import SceneClassifier
from data.dataset import FPDataset
from data.augment import Augmentations
import argparse
from get_optim import get_optimizer, get_scheduler
from classifier import Classifier
import torch.optim as optim
import optuna
import yaml
from focal_loss import FocalLoss
import json
from optuna.importance import FanovaImportanceEvaluator


def objective(trial: optuna.trial.Trial) -> float:

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cfg', type=str, default='train_fp_tuner.yaml', help='training config file')
    args = parser.parse_args()
    
    opt = args.train_cfg

    with open(opt, 'r') as config:
        opt = yaml.safe_load(config)

    parameter_tuning = opt['parameter_tuning']

    aug_tuning  = opt['aug_tuning']

    model_name = opt['model_name']
    num_classes = opt['num_classes']

    model = SceneClassifier(pretrained=False, model_name=model_name, num_classes=num_classes)

    pred_detections_pth = opt['pred_detections_pth']
    bbox_infer_dataset = opt['bbox_infer_dataset']
    annotations_pth = opt['annotations_pth']

    train_image_dir = opt['dataset']['train']['image_path']
    train_img_size = opt['dataset']['train']['img_size']

    val_image_dir = opt['dataset']['val']['image_path']
    val_img_size = opt['dataset']['val']['img_size']
    val_batch_size = opt['dataset']['val']['batch_size']
    epochs = opt['epochs']



    if parameter_tuning:

        train_batch_size = trial.suggest_int("batch_size", 32, 200)
        print('batch_size: ', train_batch_size, '\n')

        learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.003)
        print('learning_rate: ', learning_rate, '\n')

        img_size = trial.suggest_categorical('image_size',  [90, 120, 140, 160, 180])
        print('img_size: ', img_size, '\n')

        # gamma = trial.suggest_float('gamma',  2.0, 5.0)
        # alpha = trial.suggest_float('alpha',  0.25, 1.0)
        w1 = trial.suggest_float('w1', 0, 1)
        print('w1: ', w1, '\n')

        w2 = 1-w1
        print('w2: ', w2, '\n')

        #weights = torch.tensor([w1, w2])
        
        weight_decay = trial.suggest_float("weight_decay", 0, 1e-2)
        print('weight_decay: ', weight_decay, '\n')

        #weight_decay = 0
        optimizer_name = trial.suggest_categorical('optimizer_name',  ['adam', 'sgd'])
        #optimizer_name = 'adam'
        print('optimizer_name: ', optimizer_name, '\n')

    else:

        train_batch_size = opt['dataset']['train']['batch_size']

        learning_rate = opt['learning_rate']

        w1 = opt['w1']
        w2 = opt['w2']
        img_size = opt['dataset']['train']['img_size']

        gamma = opt['gamma']
        alpha = opt['alpha']

        #weight_decay = opt['weight_decay']
        weight_decay = trial.suggest_float("weight_decay", 0, 1e-2)
        optimizer_name = opt['optimizer']

    if not aug_tuning:

        train_aug_param = opt['dataset']['augmentations']

    else:
        aug_prob = trial.suggest_float("aug_prob", 0.5, 1.0)
        some_of = trial.suggest_int("some_of", 1, 3)
        #center_crop_lower = trial.suggest_float("center_crop_lower", 0.5, 0.9)
        pers = trial.suggest_float("pers", 0, 0.2)
        pers_prob = trial.suggest_float("pers_prob", 0, 1)

        srp_prob = trial.suggest_float("srp_prob", 0, 1)
        srp_down = trial.suggest_float("srp_down", 0.5, 1)
        srp_up = trial.suggest_float("srp_up", 1, 3)

        scale_down = trial.suggest_float("scale_down", 0.7, 1.0)
        scale_up = trial.suggest_float("scale_up", 1.0, 1.3)

        br_up = trial.suggest_float("brightness_up", 1, 1.2)
        br_down = trial.suggest_float("brightness_down", 0.8, 1)

        sat_up = trial.suggest_float("sat_up", 1, 1.3)
        sat_down = trial.suggest_float("sat_down", 0.7, 1)

        con_down = trial.suggest_float("con_down", 0.7, 1)
        con_up = trial.suggest_float("con_up", 1, 1.3)

        rot_up = trial.suggest_int("rot_up", 0, 30)
        rot_down = trial.suggest_int("rot_down", -30, 0)

        sh_up = trial.suggest_int("sh_up", 0, 15)
        sh_down = trial.suggest_int("sh_down", -15, 0)

        #hue_up = trial.suggest_int("hue_up", 0, 15)
        #hue_down = trial.suggest_int("hue_down", -15, 0)
        # rot_up = 0
        # rot_down = 0

        # sh_up = 0
        # sh_down = 0

        train_aug_param = {'aug_prob': aug_prob, 
                           'letterbox': False,
                            'fliplr': [0.5, 1], 
                            'resize': [img_size, img_size, 1], 
                            'perspective': [pers, pers_prob, 1],
                            'scale': [scale_down, scale_up, 1], 
                            'translate': [0.1, 0.1, 0],
                            'sharpness': [[srp_down, srp_up], srp_prob, 1],
                            'brightness': [br_down, br_up, 1], 
                            'saturation': [sat_down, sat_up, 1], 
                            'contrast': [con_down, con_up, 1], 
                            'rotate': [rot_down, rot_up, 1],
                            'hue': [0.1, 0.1, 0],
                            'add_grayscale': [0.2, 0],
                            'crop': [0.1, 0.1, 0],
                            'center_crop': [0.8, 0.8, 0],
                            'shear': [sh_down, sh_up, 1],
                            'some_of': 3}

    #optimizer_name = 'adam'
    train_augmentations = [Augmentations(train_aug_param, img_size, k=some_of)]
    val_augmentations = [Augmentations(opt['dataset']['val_augmentations'], img_size)]
    loss_name = opt['loss']
    train_dataset = FPDataset(image_path=train_image_dir,
                            augment=True,
                            augmentations=train_augmentations,
                            image_size=img_size)
    
    val_dataset = FPDataset(image_path=val_image_dir,
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
    hyperparameters = dict(learning_rate=learning_rate, batch_size=train_batch_size)

    if loss_name == 'CE':
        weights = torch.tensor([w1, w2])
        loss = nn.CrossEntropyLoss(weight=weights) 
        
    elif loss_name == 'focal':
        loss = FocalLoss(alpha=alpha, gamma=gamma)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) if optimizer_name =='adam' else \
                        optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
                        
    scheduler = get_scheduler(opt, optimizer, len(train_loader))

    classifier = Classifier(model=model, scheduler=scheduler, optimizer=optimizer, loss=loss, ckpt_path='/parameter_tuning', num_classes=num_classes, 
                            img_size=train_img_size, pred_detections_pth=pred_detections_pth,
                            annotations_pth=annotations_pth, bbox_infer_dataset=bbox_infer_dataset, 
                            model_name=model_name)
    
    trainer = pl.Trainer(gpus=1,
                        devices="auto",
                        accelerator="auto",
                         check_val_every_n_epoch=opt['val_frequency'],
                         enable_progress_bar=True,
                         max_epochs=epochs)
    trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model=classifier, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader)


    return trainer.callback_metrics["average precision"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cfg', type=str, default='train_fp_tuner.yaml', help='training config file')
    args = parser.parse_args()
    
    opt = args.train_cfg

    with open(opt, 'r') as config:
        opt = yaml.safe_load(config)


    model_name = opt['model_name']
    n_trials = opt['n_trials']
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    evaluator = FanovaImportanceEvaluator()

    # Assessing the importance of hyperparameters using the FANOVA evaluator
    param_importances = optuna.importance.get_param_importances(study, evaluator=evaluator)

    print("Hyperparameter importances:")
    for param, importance in param_importances.items():
        print(f"{param}: {importance}")

    # Optionally, save the importances to a file
    importance_path = f'hyperparameter_importances_{model_name}.json'
    with open(importance_path, 'w') as f:
        json.dump(param_importances, f, indent=4)

    optimized_parameters = {
        'best_AP': trial.value,
        'batch_size': trial.params['batch_size'],
        'learning_rate': trial.params['learning_rate'],
        'image_size': trial.params['image_size'],
        # 'gamma': trial.params['gamma'],
        # 'alpha': trial.params['alpha'],
        'w1': trial.params['w1'],
        #'w2': trial.params['w2'],
        'weight_decay': trial.params['weight_decay'],
        #'optimizer_name': trial.params['optimizer_name']
    }

    optimized_parameters = json.dumps(optimized_parameters, indent=4)
    
    with open(f'optimized_parameters_{model_name}.json', "w") as outfile:
        outfile.write(optimized_parameters)