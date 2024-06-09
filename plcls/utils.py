import torch
import torchvision
from lightning.pytorch.utilities import rank_zero_info
from torch.optim import Optimizer


def get_optimizer(conf, parameters) -> Optimizer:
    optimizer_class = getattr(torch.optim, conf.name)
    optimizer = optimizer_class(parameters, **conf.optimizer_params)
    return optimizer


def get_scheduler(conf, optimizer):
    if conf.scheduler.lower() == "onecyclelr":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=conf.optimizer.optimizer_params.lr,
            epochs=conf.trainer.trainer_params.max_epochs,
            steps_per_epoch=1,
        )
    raise RuntimeError(
        "Available schedulers: [onecyclelr].\nGot {conf.scheduler.lower()}"
    )


def get_criterion(conf):
    criterion_class = getattr(torch.nn, conf.name)
    criterion = criterion_class(**conf.criterion_params)
    return criterion


def get_model(conf):
    model = torchvision.models.get_model(**conf.model_params)
    if conf.weights:
        rank_zero_info(f"-- Loading `{conf.model_params.name}` weights.")
        weights_enum = torchvision.models.get_weight(conf.weights)
        state_dict = weights_enum.get_state_dict()

        org_state_dict = model.state_dict()
        new_state_dict = {}
        for k, v in zip(state_dict.keys(), state_dict.values()):
            org_v = org_state_dict[k]
            if v.size() == org_v.size():
                new_state_dict[k] = v
            else:
                rank_zero_info(
                    f"-- Original shape is {org_v.size()}, initializing new {v.size()}"
                )
                new_state_dict[k] = org_v

        missed, unexpected = model.load_state_dict(new_state_dict, strict=False)
        rank_zero_info(f"-- Missed keys: {missed}")
        rank_zero_info(f"-- Unexpected keys: {unexpected}")
    return model
