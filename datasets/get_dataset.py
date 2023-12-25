from datasets.tudl_db import TUDL_DB_Train, TUDL_DB_Test
from torch.utils.data import DataLoader

def get_dataset(opts, db_nm, partition, batch_size, shuffle, drop_last, cls_nm=None, n_cores=1):
    loader, db = None, None
    if db_nm == "tudl":
        if partition in ["train"]:
            db = TUDL_DB_Train(opts, partition, cls_nm)
            loader = DataLoader(db, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=n_cores, pin_memory=True)
        else:
            db = TUDL_DB_Test(opts, partition, cls_nm)
            loader = DataLoader(db, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=n_cores, pin_memory=True)
    return loader, db