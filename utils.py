import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import gc
from discriminator import Discriminator
from generator import Generator
from config import CFG

disc = Discriminator(in_channels=CFG.IMG_CHANNEL).to(CFG.DEVICE).to(CFG.DEVICE)
gen = Generator(in_channels=CFG.IMG_CHANNEL, features=64).to(CFG.DEVICE).to(CFG.DEVICE)

opt_disc = optim.Adam(disc.parameters(), lr=CFG.LR, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=CFG.LR, betas=(0.5, 0.999))

BCE = nn.BCEWithLogitsLoss()
L1_LOSS = nn.L1Loss()


def train_one_epoch(gen, disc, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler):
    pBar = tqdm(enumerate(loader), total=len(loader))

    for idx, (sketch, image) in pBar:
        sketch = sketch.to(CFG.DEVICE)
        image = image.to(CFG.DEVICE)

        # training Discriminator
        with torch.cuda.amp.autocast():
            image_fake = gen(sketch)

            D_real = disc(sketch, image)
            D_real_loss = BCE(D_real, torch.ones_like(D_real))

            D_fake = disc(sketch, image_fake.detach())
            D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))

            D_loss = D_fake_loss + D_real_loss

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Training Generator
        with torch.cuda.amp.autocast():
            D_fake = disc(sketch, image_fake)
            G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
            L1 = L1_LOSS(image_fake, image) * CFG.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        pBar.set_postfix(G_loss=f'{G_loss.item():0.4f}',
                         D_loss=f'{D_loss.item():0.4f}',
                         gpu_mem=f'{mem:0.2f} GB')
        torch.cuda.empty_cache()
        gc.collect()
