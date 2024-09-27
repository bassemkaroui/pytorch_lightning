# PYTHON_ARGCOMPLETE_OK

import argparse
import re
from pathlib import Path

import argcomplete
import lightning as L
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.tuner import Tuner
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import (MulticlassAccuracy, MulticlassF1Score,
                                         MulticlassPrecision, MulticlassRecall)
from torchvision import datasets, models, transforms

ROOT_DIR = Path.home() / "mnist"  # Path.cwd()
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
BATCH_SIZE = 256  # 464


class MNISTDataModule(L.LightningDataModule):

    def __init__(self, data_dir: str | Path = "./", batch_size: int = 64) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size  # important for batch size finder
        self._MNIST_STATS = (0.1307,), (0.3081,)
        self._img_size = (224, 224)
        self.prepare_data_per_node = True
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(self._img_size),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(*self.MNIST_STATS),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ]
        )
        self.test_transforms = transforms.Compose(
            [
                transforms.Resize(self._img_size),
                transforms.ToTensor(),
                transforms.Normalize(*self.MNIST_STATS),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ]
        )

    @property
    def MNIST_STATS(self):
        return self._MNIST_STATS

    def prepare_data(self) -> None:
        # for downloading and preprocessing, without setting any state (ie: self.x = y)
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            mnist = datasets.MNIST(
                self.data_dir, train=True, transform=self.train_transforms
            )
            self.num_classes = len(mnist.classes)
            self.train_ds, self.val_ds = random_split(
                mnist, [55000, 5000], generator=torch.Generator().manual_seed(314)
            )
        if stage == "test":
            self.test_ds = datasets.MNIST(
                self.data_dir, train=False, transform=self.test_transforms
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, self.batch_size, num_workers=4, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, self.batch_size, num_workers=4, pin_memory=True)


# Lightning module
class MNISTModel(pl.LightningModule):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self._fine_tuning = False
        self.resnet.requires_grad_(False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        metrics = torchmetrics.MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes),
                "recall": MulticlassRecall(num_classes),
                "precision": MulticlassPrecision(num_classes),
                "f1_score": MulticlassF1Score(num_classes),
            },
            compute_groups=[["accuracy", "recall", "precision", "f1_score"]],
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.learning_rate = 1e-3  # important for learning rate finder

    @property
    def fine_tuning(self) -> bool:
        return self._fine_tuning

    @fine_tuning.setter
    def fine_tuning(self, value: bool) -> None:
        if value == self._fine_tuning:
            return
        self._fine_tuning = value
        if value:
            for name, layer in self.resnet.named_children():
                if name in ("layer4", "fc"):
                    layer.requires_grad_()
        else:
            self.resnet.requires_grad_(False)
            self.resnet.fc.requires_grad_()

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self(imgs)
        loss = F.cross_entropy(preds, targets)
        self.train_metrics.update(preds, targets)
        for metric_name, metric in self.train_metrics.items():
            self.log(metric_name, metric, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    @pl.utilities.rank_zero_only
    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self(imgs)
        loss = F.cross_entropy(preds, targets)
        self.val_metrics.update(preds, targets)
        for metric_name, metric in self.val_metrics.items():
            self.log(metric_name, metric, on_step=False, on_epoch=True)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.resnet.fc.parameters(), "lr": self.learning_rate},
                {
                    "params": self.resnet.layer4.parameters(),
                    "lr": self.learning_rate / 2,
                },
            ],
            amsgrad=True,
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }


def pos_int(value: str):
    int_value = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError(
            f"Invalid value {value}: must be a positive number"
        )
    return int_value


def device_type(value: str):
    if re.match(r"-?\d+", value):
        int_value = int(value)
        if int_value == 0 or int_value < -1:
            raise argparse.ArgumentTypeError(
                "Must be an integer greater than 0 or equal to -1"
            )
        return int_value
    elif value == "auto":
        return value
    elif re.match(r"\[(\d+,\s)*\d\]", value):
        return eval(value)


def main():
    parser = argparse.ArgumentParser(
        description="Pytorch Lightning Trainer for MNIST dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--batch-size", type=pos_int, default=BATCH_SIZE, help="-")
    parser.add_argument("--data-dir", default=DATA_DIR, help="-")
    parser.add_argument("--fine-tuning", action="store_true", help="-")
    parser.add_argument("--early-stopping", action="store_true", help="-")
    parser.add_argument("--lr-monitoring", action="store_true", help="-")
    parser.add_argument("--checkpointing", action="store_true", help="-")
    parser.add_argument("--lr-finder", action="store_true", help="-")
    parser.add_argument("--batch-size-finder", action="store_true", help="-")

    parser.add_argument("--num-sanity-val-steps", type=int, default=2, help="-")
    parser.add_argument("--default-root-dir", default=LOG_DIR, help="-")
    parser.add_argument("--gradient-clip-val", type=float, default=0.5, help="-")
    parser.add_argument("--accumulate-grad-batches", type=pos_int, default=2, help="-")
    parser.add_argument("--max-epochs", type=pos_int, default=5, help="-")
    parser.add_argument(
        "--precision", choices=["16-mixed", "32-true"], default="16-mixed", help="-"
    )
    parser.add_argument(
        "--strategy", choices=("ddp", "dp", "auto"), default="auto", help="-"
    )
    parser.add_argument("--num-nodes", type=pos_int, default=1, help="-")
    parser.add_argument("--devices", type=device_type, default="auto", help="-")
    parser.add_argument(
        "--accelerator", choices=("gpu", "cpu", "auto"), default="cpu", help="-"
    )

    argcomplete.autocomplete(parser)
    args, _ = parser.parse_known_args()

    mnist_dm = MNISTDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    model = MNISTModel()

    callbacks = []
    if args.early_stopping:
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, verbose=True
        )
        callbacks.append(early_stopping)

    if args.lr_monitoring:
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    trainer_args = [
        "accelerator",
        "devices",
        "num_nodes",
        "strategy",
        "precision",
        "max_epochs",
        "accumulate_grad_batches",
        "gradient_clip_val",
        "default_root_dir",
        "num_sanity_val_steps",
    ]

    training_config = {
        attr_name: getattr(args, attr_name) for attr_name in trainer_args
    }
    training_config["callbacks"] = []
    if args.checkpointing:
        checkpointing = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filename="{epoch:02d}-{step}-{val_loss:.3f}",
            mode="min",
        )
        training_config["callbacks"].append(checkpointing)

    # training_config = {
    #     "accelerator": "gpu",
    #     "devices": 1,
    #     # "num_nodes": 2,
    #     # "strategy": "ddp",
    #     "precision": "16-mixed",
    #     "max_epochs": 5,
    #     "accumulate_grad_batches": 2,
    #     "gradient_clip_val": 0.1,
    #     "default_root_dir": LOG_DIR,
    #     # "num_sanity_val_steps": 0,
    #     "callbacks": [
    #         checkpointing,
    #         # pl.callbacks.LearningRateFinder(),
    #         # pl.callbacks.BatchSizeFinder(mode="binsearch", init_val=8),
    #     ],
    # }

    trainer = pl.Trainer(**training_config)

    tuner = Tuner(trainer)
    if args.batch_size_finder:
        tuner.scale_batch_size(model, datamodule=mnist_dm, mode="binsearch", init_val=8)
    if args.lr_finder:
        tuner.lr_find(model, datamodule=mnist_dm)

    trainer.callbacks.extend(callbacks)

    trainer.fit(model, datamodule=mnist_dm)

    # Fine tuning
    if args.fine_tuning:
        model.fine_tuning = args.fine_tuning

        training_config["max_epochs"] = 30
        trainer = pl.Trainer(**training_config)
        tuner = Tuner(trainer)
        if args.batch_size_finder:
            tuner.scale_batch_size(
                model, datamodule=mnist_dm, mode="binsearch", init_val=8
            )
        if args.lr_finder:
            tuner.lr_find(model, datamodule=mnist_dm)

        trainer.callbacks.extend(callbacks)

        trainer.fit(model, datamodule=mnist_dm)


if __name__ == "__main__":
    main()
