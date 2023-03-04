import argparse
import logging
import random
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import numpy as np

from dataset import DSB2018Dataset
from utils import ext_transforms as et
from utils.dice_score import dice_loss
from evaluate import evaluate
from predict import predict
from unet_diff.model import UNet_with_IELs

image_dir = './data/combined'
mask_dir = './data/combined'
test_dir = './data/testing_data'
dir_predictions = './predictions'
dir_checkpoint = './checkpoints/'

def train_net(net,
              device,
              epochs: int = 1,
              batch_size: int = 1,
              learning_rate: float = 1*1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              amp: bool = False,
              add_noise: bool = False,
              use_iels: bool = False):
    # 1. Setup random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    # 2. Create dataset
    if add_noise:
      train_transform = et.ExtCompose([
      et.ExtResize(size=(256, 256)),
      et.add_noise_to_lbl(num_classes=2, scale=3, keep_prop=0.9),
      et.ExtToTensor(normalize=False)
      ])
    else:
      train_transform = et.ExtCompose([
      et.ExtResize(size=(256, 256)),
      et.ExtToTensor(normalize=False)
      ])
      
    val_transform = et.ExtCompose([
    et.ExtResize(size=(256, 256)),
    et.ExtToTensor(normalize=False)
    ])
    
    dataset_train = DSB2018Dataset(image_dir, mask_dir, train=True, transform=train_transform)
    dataset_val = DSB2018Dataset(image_dir, mask_dir, train=True, transform=val_transform)
    n_val = int(len(dataset_train) * val_percent)
    n_train = len(dataset_train) - n_val
    train_set, _ = random_split(dataset_train, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    _, val_set = random_split(dataset_val, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # # (Initialize logging)
    experiment = wandb.init(project='DSB2018', resume='allow', entity="cyliu111")
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                 val_percent=val_percent, save_checkpoint=save_checkpoint,
                                 amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'] 

                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images, use_iels)
                    loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                   'train loss': loss.item(),
                   'step': global_step,
                   'epoch': epoch
                })
                pbar.set_postfix(**{'train loss (batch)': loss.item()})

                # Evaluation round
                if epoch % 2 == 0:
                    division_step = (n_train // (1 * batch_size))
                    if global_step % division_step == 0:
                        val_score = evaluate(net, val_loader, device, False)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                           'learning rate': optimizer.param_groups[0]['lr'],
                           'validation Dice': val_score,
                           'images': wandb.Image(images[0].cpu()),
#                            'masks': {
#                                'true': wandb.Image(true_masks[0].float().cpu()),
#                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
#                            },
                           'step': global_step,
                           'epoch': epoch,
                        })
        
#         if epoch == epochs:
#             for batch in val_loader: 
#                 images = batch['image']
#                 true_masks = batch['mask']
#                 masks_pred, dice_score = predict(net, batch, device, False)

            
#             if dice_score < 0.96:
#                 for i in range(10):
#                     experiment.log({'bad_dice': dice_score,
#                         'bad prediction': wandb.Image(images[i].cpu()),
#                         'bad_true': wandb.Image(true_masks[i].float().cpu()),
#                         'bad_pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[i].float().cpu())
#                         }
#                         ) 
          
        if epoch == epochs:
            Path(dir_predictions).mkdir(parents=True, exist_ok=True)
            for i, batch in enumerate(val_loader):
                images, true_masks = batch['image'], batch['mask']
                masks_pred, dice_score = predict(net, batch, device, False)

                image = np.array(images[0].cpu(), dtype=np.uint8).transpose(1,2,0)
                mask = np.array(true_masks[0].cpu(), dtype=np.uint8)*255
                pred = np.array(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].cpu(), dtype=np.uint8)*255

                Image.fromarray(image).save(str(Path(dir_predictions)/'{}_image.png'.format(i)))
                Image.fromarray(mask).save(str(Path(dir_predictions)/'{}_mask.png'.format(i)))
                Image.fromarray(pred).save(str(Path(dir_predictions)/'{}_pred.png'.format(i)))
  
        if epoch%50 == 0:
          if save_checkpoint:
              Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
              torch.save(net.state_dict(), str(Path(dir_checkpoint) / 'checkpoint_epoch{}.pth'.format(epoch)))
              logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--add_noise', action='store_true', default=False, help='Add noise to labels')
    parser.add_argument('--use_iels', action='store_true', default=False, help='Use IELs for training')
    parser.add_argument('--iels_dt', '-dt', type=float, default=0.1, help='dt for IELs')
    parser.add_argument('--iels_num', '-num', type=int, default=30, help='Number of IELs')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet_with_IELs(n_channels=3, n_classes=args.classes, bilinear=args.bilinear, iels_dt=args.iels_dt,
                         iels_num=args.iels_num)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    
    if args.use_iels:
        logging.info(f'Use IELs for training:\n'
                     f'\t iels_dt = {net.iels_dt} \n'
                     f'\t iels_num = {net.iels_num}')
    else:
        logging.info(f'No IELs for training')
        
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  add_noise=args.add_noise,
                  use_iels=args.use_iels)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise


