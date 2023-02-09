import click
from os.path import join, dirname, abspath
import subprocess
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import yaml
import self_supervised_agrinet.datasets.datasets as datasets
import self_supervised_agrinet.models.models as models



@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)

def main(config,weights,checkpoint):
    cfg = yaml.safe_load(open(config))
    # save the version of git we're using 
    cfg['git_commit_version'] = str(subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).strip())

    # Load data and model
    data = datasets.StatDataModule(cfg)    
    if weights is None:
        model = models.BarlowTwins(cfg)
    else:
        model = models.BarlowTwins.load_from_checkpoint(weights,hparams=cfg)
        
    # Add callbacks:
    #lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(monitor='train:loss',
                                 save_top_k=5,
                                 every_n_train_steps = 10000,
                                 filename=cfg['experiment']['id']+'_{epoch:02d}_{loss:.2f}',
                                 mode='min',
                                 save_last=True)


    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                             default_hp_metric=False)

    # Setup trainer
    trainer = Trainer(gpus=cfg['train']['n_gpus'],
                      logger=tb_logger,
                      resume_from_checkpoint=checkpoint,
                      max_epochs= cfg['train']['max_epoch'],
                      callbacks=[checkpoint_saver])
    # Train
    trainer.fit(model, data)

if __name__ == "__main__":
    main()
