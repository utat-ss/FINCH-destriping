# stdlib
import argparse
import os

# external
import lpips
import pytorch_lightning as pl
import scipy.io as sio
import torch
import torchmetrics
import torchvision.transforms.functional as TF
from model import SERT
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor


class HyperspectralDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.files = os.listdir(directory)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.files[idx])
        data = sio.loadmat(file_path)
        input_image = data["input"]
        gt_image = data["gt"]

        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)

        return input_image, gt_image


class SERTLightningModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.model = SERT(
            inp_channels=31,
            dim=90,
            window_sizes=[8, 8, 8, 8, 8, 8],
            depths=[6, 6, 6, 6, 6, 6],
            num_heads=[6, 6, 6, 6, 6, 6],
            split_sizes=[1, 1, 1, 1, 1, 1],
            mlp_ratio=2,
            down_rank=16,
            memory_blocks=256,
            qkv_bias=True,
            qk_scale=None,
            bias=False,
            drop_path_rate=0.1,
            weight_factor=0.1,
        )
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.MSELoss()

        self.train_psnr = torchmetrics.PSNR(data_range=1.0)
        self.train_ssim = torchmetrics.SSIM(data_range=1.0)
        self.lpips_fn = lpips.LPIPS(net="alex")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)

        # Calculate metrics
        psnr = self.train_psnr(outputs, targets)
        ssim = self.train_ssim(outputs, targets)
        lpips_value = self.lpips_fn(
            self._normalize(outputs), self._normalize(targets)
        ).mean()

        self.log("train_loss", loss)
        self.log("train_psnr", psnr, on_epoch=True, prog_bar=True)
        self.log("train_ssim", ssim, on_epoch=True, prog_bar=True)
        self.log("train_lpips", lpips_value, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _normalize(self, img):
        return TF.normalize(img, mean=[0.5], std=[0.5])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the SERT model for Hyperspectral Image Denoising"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing .mat files for training",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    return parser.parse_args()


def main():
    args = parse_args()

    train_dataset = HyperspectralDataset(directory=args.data_dir, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = SERTLightningModel(learning_rate=args.learning_rate)

    wandb_logger = WandbLogger(project="HyperspectralDenoising", log_model="all")
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss", mode="min", save_top_k=3
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
