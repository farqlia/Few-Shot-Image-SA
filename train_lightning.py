from fsimgsa.model.base_model import BaseModel, ModelLightning
from fsimgsa.data.fer2013_dm import FER2013DataModule
from fsimgsa.data.jaffe_dm import JaffeDataModule
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


@hydra.main(config_path="./configs", config_name="base_model_config.yaml", version_base="1.2")
def train_base_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)

    dm = JaffeDataModule(
        transform_cfg=cfg.transform,
        data_path=cfg.dataset.data_path,
        batch_size=cfg.train.batch_size,
        shuffle=cfg.train.shuffle,
        num_workers=cfg.train.num_workers,
        val_split=cfg.dataset.val_split,
    )
    dm.setup()

    model = BaseModel(n_classes=dm.num_classes)
    pl_model = ModelLightning(model, dm.classes)

    model_save_dir = cfg.model.save_dir
    model_name = cfg.dataset.name
    logger = TensorBoardLogger(save_dir=model_save_dir, name=model_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_dir,
        filename=model_name + "-{epoch:02d}-{val_loss:.2f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_callback, lr_monitor]

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs, 
        accelerator=cfg.device, 
        callbacks=callbacks, 
        logger=logger
    )

    trainer.fit(pl_model, datamodule=dm)
    trainer.test(pl_model, datamodule=dm)

if __name__ == "__main__":
    train_base_model()
