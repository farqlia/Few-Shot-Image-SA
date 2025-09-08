from typing import List
import torch.nn as nn
import pytorch_lightning as pl 
import torchmetrics
import torch 
import matplotlib.pyplot as plt
import seaborn as sns

class ModelLightning(pl.LightningModule):

    def __init__(self, model: nn.Module, classes: List):
        super().__init__()
        self.model = model 
        self.class_names = classes
        self.num_classes = len(classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        
        self.val_preds = []
        self.val_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.train_acc(y_hat, y)
        self.log('train/loss', loss, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_epoch=True, prog_bar=True)
        return loss 
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.val_acc(y_hat, y)
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.val_preds.append(y_hat.cpu())
        self.val_targets.append(y.cpu())
        return loss
    
    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)

        cm = torchmetrics.functional.confusion_matrix(
            preds, targets, task="multiclass", num_classes=self.num_classes
        ).cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (Epoch {self.current_epoch})")

        self.logger.experiment.add_figure("val/confusion_matrix", fig, self.current_epoch)
        plt.close(fig)

        self.val_preds.clear()
        self.val_targets.clear()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.test_acc(y_hat, y)
        self.log('test/loss', loss, on_epoch=True, prog_bar=True)
        self.log('test/acc', self.test_acc, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

class BaseModel(nn.Module):
    def __init__(self, n_classes):
        super(BaseModel, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, n_classes)


    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    