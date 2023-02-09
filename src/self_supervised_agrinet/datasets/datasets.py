import torch
import yaml
import torchvision
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import os
import os.path as path
import splitfolders
from PIL import Image, ImageFile
import self_supervised_agrinet.utils.utils as utils

ImageFile.LOAD_TRUNCATED_IMAGES = True

class StatDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        # from cfg i can access to all my shit
        # as data path, data size and so on 
        self.cfg = cfg
        self.len = -1
        self.setup()
        self.loader = [ self.train_dataloader() ]

    def prepare_data(self):
        # Augmentations are applied using self.transform 
        # no data to download, for now everything is local 
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train = SugarBeets(self.cfg['data']['path'], self.cfg['train']['mode'])
        return

    def train_dataloader(self):
        loader = DataLoader(self.data_train, 
                            batch_size = self.cfg['train']['batch_size'] // self.cfg['train']['n_gpus'],
                            num_workers = self.cfg['train']['workers'],
                            shuffle=True)
        self.len = self.data_train.__len__()
        return loader
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass


#################################################
################## Data loader ##################
#################################################

class SugarBeets(Dataset):
    def __init__(self, datapath, mode):
        super().__init__()
        
        self.datapath = datapath

        with open(self.datapath + "/split.yaml") as f:
            self.all_data = yaml.safe_load(f)
        
        if mode == 2:
            self.all_imgs = self.all_data['pre-training'][0:4000]
        else:
            self.all_imgs = self.all_data['pre-training']
        self.len = len(self.all_imgs)
        print(f'I am loading {self.len} files as a dataset.')
        self.transform = utils.Transform()

    def __getitem__(self, index):
        img_loc = os.path.join(self.datapath, self.all_imgs[index])
        img = Image.open(img_loc).convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor

    def __len__(self):
        return self.len

