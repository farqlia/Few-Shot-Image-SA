from fsimgsa.data.fer2013 import get_fer2013_transform, FER2013Dataset
from fsimgsa.data.dir_split_dm import Dir_Split_DataModule


class FER2013DataModule(Dir_Split_DataModule):
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
            transform = get_fer2013_transform(transform_cfg.fer2013),
            data_set_cls = FER2013Dataset
        )