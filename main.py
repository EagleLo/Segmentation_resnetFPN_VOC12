import argparse
import os
import wandb

from torchvision import transforms
from torch.utils.data import DataLoader

from datasets.voc import VOC
from trainer import Trainer

# try:
#     from nsml import DATASET_PATH
#     path = DATASET_PATH
# except ImportError:
path = '/content/'

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_loader(config):
    if config.dataset == 'voc':
        root = os.path.join(path, 'train')
        transform = transforms.Compose([
                transforms.Pad(10),
                transforms.CenterCrop((config.h_image_size, config.w_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        train_data_set = VOC(root=root,
                             image_size=(config.h_image_size, config.w_image_size),
                             dataset_type='train',
                             transform=transform)
        train_data_loader = DataLoader(train_data_set,
                                       batch_size=config.train_batch_size,
                                       shuffle=True)

        val_data_set = VOC(root=root,
                           image_size=(config.h_image_size, config.w_image_size),
                           dataset_type='val',
                           transform=transform)
        val_data_loader = DataLoader(train_data_set,
                                     batch_size=config.val_batch_size,
                                     shuffle=False) # For make samples out of various models, shuffle=False
    elif config.dataset == 'gta':
        # TODO:
        pass

    return train_data_loader, val_data_loader


def main(config):
    import sys
    print(sys.version)
    make_dir(config.model_save_path)
    make_dir(config.sample_save_path)

    if config.mode == 'train':
        train_data_loader, val_data_loader = get_loader(config)
        trainer = Trainer(train_data_loader=train_data_loader,
                         val_data_loader=val_data_loader,
                         config=config)
        trainer.train_val()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #
    parser.add_argument('--mode', type=str, default='train', choices=['train'])
    parser.add_argument('--model', type=str, default='fcn8', choices=['fcn8', 'unet', 'pspnet_avg',
                                                                      'pspnet_max', 'dfnet'])
    parser.add_argument('--dataset', type=str, default='voc', choices=['voc'])


    # Training setting
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_exp', type=float, default=0.9)
    parser.add_argument('--beta1', type=float, default=5e-1)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--h_image_size', type=int, default=512)
    parser.add_argument('--w_image_size', type=int, default=256)
    # Hyper parameters
    #TODO

    # Path
    parser.add_argument('--model_save_path', type=str, default='/content/model_single')
    parser.add_argument('--sample_save_path', type=str, default='/content/samples')

    # Logging setting
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--val_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10, help='Saving epoch')
    parser.add_argument('--sample_save_step', type=int, default=10, help='Saving epoch')

    #Weight & Biases init:
    wandb.init(
      # Set the project where this run will be logged
      project="preonboarding-Caltech",
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"segmentation_0",
      # Track hyperparameters and run metadata
      config={
      "lr": 0.05,
      "architecture": "Unet",
      "dataset": "VOC12",
      "epochs": 3,
      "iterations" : 15
      })

    # MISC
    config = parser.parse_args()
    print(config)
    main(config)
