import lightning as L
from torchmetrics.classification import MulticlassAccuracy

from plcls.utils import get_criterion, get_model, get_optimizer, get_scheduler


class ClassifierModel(L.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.model = get_model(conf.model)
        self.criterion = get_criterion(conf.criterion)

        num_classes = conf.model.model_params.num_classes
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        del batch_idx
        image, label = batch

        pred = self.model(image)
        train_loss = self.criterion(pred, label)
        self.train_accuracy(pred, label)

        self.log("train/accuracy", self.train_accuracy, on_step=True, on_epoch=True)
        self.log("train/loss", train_loss, on_step=True, on_epoch=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        image, label = batch

        pred = self.model(image)
        valid_loss = self.criterion(pred, label)
        self.val_accuracy(pred, label)

        self.log("valid/accuracy", self.val_accuracy, on_step=True, on_epoch=True)
        self.log("valid/loss", valid_loss, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        parameters = self.model.parameters()
        optimizer = get_optimizer(self.conf.optimizer, parameters)
        scheduler = get_scheduler(self.conf, optimizer)
        return [optimizer], [scheduler]
