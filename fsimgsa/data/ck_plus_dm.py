from fsimgsa.data.ck_plus import get_ck_plus_transform, CKPlusDataset
from fsimgsa.data.dir_split_dm import Dir_Split_DataModule

class CKPlusDataModule(Dir_Split_DataModule):
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
            transform = get_ck_plus_transform(transform_cfg.ckplus),
            data_set_cls = CKPlusDataset
        )