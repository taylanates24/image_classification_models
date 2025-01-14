import torch.optim as optim
import math

def get_optimizer(opt, model, wd=0):
    if opt['optimizer'] == 'adam':
        return optim.Adam(model.parameters(), lr=opt['learning_rate'])
    elif opt['optimizer'] == 'adamw':
        return optim.AdamW(model.parameters(), lr=opt['learning_rate'], weight_decay=wd)
    elif opt['optimizer'] == 'sgd':
        return optim.SGD(model.parameters(), lr=opt['learning_rate'], momentum=0.9)
    elif opt['optimizer'] == 'asgd':
        return optim.ASGD(model.parameters(), lr=opt['learning_rate'])
    
    
def get_scheduler(opt, optimizer, len_dataset):

    if opt['lr_scheduler'] == 'multistep_lr':
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(opt['epochs']*0.7), int(opt['epochs']*0.9)], gamma=0.1)
    elif opt['lr_scheduler'] == 'cosine':
        lf = lambda x: (((1 + math.cos(x * math.pi / 30)) / 2) ** 1.0) * 0.95 + 0.05
        #lf = lambda x: (((1 + math.cos(x * math.pi / opt['epochs'])) / 2) ** 1.0) * 0.95 + 0.05
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    elif opt['lr_scheduler'] == 'cosine_annealing':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, len_dataset, eta_min=0)