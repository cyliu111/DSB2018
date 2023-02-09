import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import numpy as np
from scipy import ndimage

from dataset import DSB2018Dataset
from utils import ext_transforms as et
from utils.dice_score import dice_loss
from evaluate import evaluate
from predict import predict
from unet_diff.model import AVEUNet

image_dir = '/content/DSB2018/data/combined'
mask_dir = '/content/DSB2018/data/combined'
test_dir = '/content/DSB2018/data/testing_data'
results_dir = '/content/DSB2018/data/results'

def train_net(net,
              device,
              epochs: int = 1,
              batch_size: int = 1,
              learning_rate: float = 1*1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 1.0,
              amp: bool = False):
    # 1. Create dataset
    train_transform = et.ExtCompose([
    # et.ExtRandomCrop(size=(256, 256)),
    et.ExtResize(256),
    # et.add_noise_to_lbl(num_classes=opts.num_classes, scale=5, keep_prop=0.9),
    et.ExtToTensor(),
])
    val_transform = et.ExtCompose([
    # et.ExtRandomCrop(size=(256, 256)),
    et.ExtResize(256),
    # et.add_noise_to_lbl(num_classes=opts.num_classes, scale=5, keep_prop=0.9),
    et.ExtToTensor(),
])
    dataset_train = DSB2018Dataset(image_dir, mask_dir, train=True)
    n_val = int(len(dataset_train) * val_percent)
    n_train = len(dataset_train) - n_val
    train_set, val_set = random_split(dataset_train, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_set.transform = train_transform
    val_set.transform = val_transform

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # # (Initialize logging)
    experiment = wandb.init(project='DSB2018', resume='allow', entity="cyliu111")
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                 val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                 amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    global_step = 0

    # wandb.watch(net, log="all")

    val_score_old = 100
    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'] 
                # images = images + 50*torch.randn(images.shape)

                true_masks = batch['mask']
                # true_masks = batch['mask_noise']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images, False)
                    loss = criterion(masks_pred, true_masks)
                    # loss = criterion(masks_pred, true_masks) \
                    #        + dice_loss(F.softmax(masks_pred, dim=1).float(),
                    #                    F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                    #                    multiclass=True)

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
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                if epoch % 50 == 0:
                    division_step = (n_train // (1 * batch_size))
                    if global_step % division_step == 0:
                        val_score = evaluate(net, val_loader, device, False)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                           'learning rate': optimizer.param_groups[0]['lr'],
                           'validation Dice': val_score,
                           'images': wandb.Image(images[0].cpu()),
                           'masks': {
                               'true': wandb.Image(true_masks[0].float().cpu()),
                               'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                           },
                           'step': global_step,
                           'epoch': epoch,
                           # **histograms
                        })
        
        if epoch == epochs:
          for batch in val_loader: 
            images = batch['image']
            true_masks = batch['mask']
            masks_pred, dice_score = predict(net, batch, device, False)
            masks_pred_ave, dice_score_ave = predict(net, batch, device, True)

            if dice_score < 0.9:
              thresh = 0.1
              experiment.log({'bad_dice': dice_score,
                      'bad_dice_ave': dice_score_ave,
                      'bad prediction': wandb.Image(images[0].cpu()),
                      'bad_true': wandb.Image(true_masks[0].float().cpu()),
                      'bad_pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                      'bad_pred_ave': wandb.Image(torch.softmax(masks_pred_ave, dim=1).argmax(dim=1)[0].float().cpu()),
                      'background': wandb.Image((masks_pred[0,0,:,:]>thresh).float().cpu()),
                      'part1': wandb.Image((masks_pred[0,1,:,:]>thresh).float().cpu()),
                      }
                      )                               
        
        if epoch%10 == 0:
          if save_checkpoint:
              Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
              torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
              logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
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

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = AVEUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

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
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise

