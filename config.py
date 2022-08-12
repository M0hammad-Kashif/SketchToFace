import torch


class CFG:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR = 1e-3
    BATCH_SIZE = 32
    IMG_CHANNEL = 3
    L1_LAMBDA = 100
    LAMBDA_GP = 10
    EPOCHS = 500
    IMG_SIZE = 256

