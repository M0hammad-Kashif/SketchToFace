import torch
import wandb
from torch.utils.data import dataloader

from config import CFG
from utils import gen, disc, opt_disc, opt_gen, L1_LOSS, BCE, train_one_epoch


def train(CFG=CFG, loader=dataloader):
    wandb.init("Sketch2Face")
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    # start = time.time()

    for epoch in range(CFG.EPOCHS):
        print(f"Epoch {epoch + 1}")

        train_one_epoch(gen, disc, dataloader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler)
        print("-" * 100)
        print("\n")


train()

