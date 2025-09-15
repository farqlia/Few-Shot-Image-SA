from fsimgsa.data.jaffe import get_jaffe_transform, JaffeDataset
from fsimgsa.data.dir_split_dm import Dir_Split_DataModule


class JaffeDataModule(Dir_Split_DataModule):
    def __init__(
        self,
        transform_cfg: dict,
        data_path: str,
        batch_size: int,
        val_split: float,
        shuffle: bool,
        num_workers: int,
    ):
        super().__init__(
            transform_cfg,
            data_path,
            batch_size,
            val_split,
            shuffle,
            num_workers,
            transform = get_jaffe_transform(transform_cfg.jaffe),
            data_set_cls = JaffeDataset
        )