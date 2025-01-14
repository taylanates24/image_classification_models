import torch.optim as optim
import math
from enum import Enum

class LRSchedulerEnum(Enum):
    COSINE = "cosine"
    MULTISTEP_LR = "multistep_lr"
    COSINE_ANNEALING = "cosine_annealing"

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

def get_optimizer(opt, model, wd=0):
    if opt['optimizer'] == 'adam':
        return optim.Adam(model.parameters(), lr=opt['learning_rate'])
    elif opt['optimizer'] == 'adamw':
        return optim.AdamW(model.parameters(), lr=opt['learning_rate'], weight_decay=wd)
    elif opt['optimizer'] == 'sgd':
        return optim.SGD(model.parameters(), lr=opt['learning_rate'], momentum=0.9)
    elif opt['optimizer'] == 'asgd':
        return optim.ASGD(model.parameters(), lr=opt['learning_rate'])
    
    
def get_scheduler(scheduler_name, optimizer, len_dataset, num_epochs):
    """
    Get the learning rate scheduler based on the specified scheduler name.

    Args:
        scheduler_name (LRSchedulerEnum): The name of the scheduler (enum value).
        optimizer (Optimizer): The optimizer instance.
        len_dataset (int): The length of the dataset (number of steps per epoch).
        num_epochs (int): Total number of epochs for training.

    Returns:
        Scheduler: An instance of the selected scheduler.
    """
    if scheduler_name == LRSchedulerEnum.MULTISTEP_LR:
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(num_epochs * 0.7), int(num_epochs * 0.9)], gamma=0.1
        )
    elif scheduler_name == LRSchedulerEnum.COSINE:
        lf = lambda x: (((1 + math.cos(x * math.pi / num_epochs)) / 2) ** 1.0) * 0.95 + 0.05
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    elif scheduler_name == LRSchedulerEnum.COSINE_ANNEALING:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len_dataset, eta_min=0)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
