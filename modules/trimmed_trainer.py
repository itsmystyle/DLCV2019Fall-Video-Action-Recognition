import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataset import TrimmedVideosDataset
from models.SCNN import SequentialCNN
from metrics import MulticlassAccuracy
from utils import set_random_seed


class TrimmedTrainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        writer,
        metric,
        save_dir,
        device,
        accumulate_gradient=1,
    ):
        # prepare model and optimizer
        self.model = model.to(device)
        self.optimizer = optimizer

        # prepare loss
        self.criterion = criterion

        # prepare dataset
        self.train_loader = train_loader
        self.val_loader = val_loader

        # parameters
        self.accumulate_gradient = accumulate_gradient

        # utils
        self.device = device
        self.writer = writer
        self.metric = metric
        self.save_dir = save_dir

    def fit(self, epochs):
        print("===> start training ...")
        iters = -1
        val_iters = -1
        best_accuracy = 0.0

        for epoch in range(1, epochs + 1):
            loss, iters = self._run_one_epoch(epoch, iters)
            val_loss, best_accuracy, val_iters = self._eval_one_epoch(
                val_iters, best_accuracy
            )

    def _run_one_epoch(self, epoch, iters):
        self.model.train()

        trange = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="Epoch {}".format(epoch),
        )

        self.metric.reset()
        batch_loss = 0.0

        for idx, batch in trange:
            iters += 1

            frames = batch["frames"].to(self.device)
            frames_len = batch["frames_len"].to(self.device)
            labels = batch["labels"].to(self.device)

            preds = self.model(frames, frames_len)

            # calculate loss and update weights
            loss = self.criterion(preds, labels)
            if idx % self.accumulate_gradient == 0:
                self.optimizer.zero_grad()
            loss.backward()
            if (idx + 1) % self.accumulate_gradient == 0:
                self.optimizer.step()

            # update metric
            self.metric.update(preds, labels)

            # update loss
            batch_loss += loss.item()
            self.writer.add_scalars(
                "Loss",
                {"iter_loss": loss.item(), "avg_loss": batch_loss / (idx + 1)},
                iters,
            )

            # update tqdm
            trange.set_postfix(
                loss=batch_loss / (idx + 1),
                **{self.metric.name: self.metric.print_score()}
            )

        if (idx + 1) % self.accumulate_gradient != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return batch_loss / (idx + 1), iters

    def _eval_one_epoch(self, val_iters, best_accuracy):
        self.model.eval()

        trange = tqdm(
            enumerate(self.val_loader), total=len(self.val_loader), desc="Valid"
        )

        self.metric.reset()
        batch_loss = 0.0

        with torch.no_grad():
            for idx, batch in trange:
                val_iters += 1

                frames = batch["frames"].to(self.device)
                frames_len = batch["frames_len"].to(self.device)
                labels = batch["labels"].to(self.device)

                preds = self.model(frames, frames_len)
                loss = self.criterion(preds, labels)

                # update loss
                batch_loss += loss.item()
                self.writer.add_scalars(
                    "Val_Loss",
                    {"iter_loss": loss.item(), "avg_loss": batch_loss / (idx + 1)},
                    val_iters,
                )

                # update metric
                self.metric.update(preds, labels)

                # update tqdm
                trange.set_postfix(
                    loss=batch_loss / (idx + 1),
                    **{self.metric.name: self.metric.print_score()}
                )

            # save best acc model
            if self.metric.get_score() > best_accuracy:
                print("Best model saved!")
                self.save(os.path.join(self.save_dir, "model_best.pth.tar"))
                best_accuracy = self.metric.get_score()

        return batch_loss / (idx + 1), best_accuracy, val_iters

    def save(self, path):
        torch.save(
            self.model.state_dict(), path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sequential trimmed video.")
    parser.add_argument(
        "model", type=str, default="SCNN", help="Which model to train (SCNN or RCNN)."
    )
    parser.add_argument("data_path", type=str, help="Path to data directory.")
    parser.add_argument("save_dir", type=str, help="Where to save trained model.")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-6, help="Weight decay rate."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--max_padding", type=int, default=24, help="Max padding length of frames."
    )
    parser.add_argument(
        "--n_workers", type=int, default=8, help="Number of worker for dataloader."
    )
    parser.add_argument(
        "--accumulate_gradient",
        type=int,
        default=1,
        help="Accumulate gradients before updating the weight.",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    set_random_seed(args.random_seed)
    writer = SummaryWriter(os.path.join(args.save_dir, "train_logs"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # prepare dataset
    train_datapath = os.path.join(args.data_path, "video", "train")
    train_labelpath = os.path.join(args.data_path, "label", "gt_train.csv")
    train_dataset = TrimmedVideosDataset(
        train_datapath, train_labelpath, max_padding=args.max_padding
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        collate_fn=train_dataset.collate_fn,
    )

    val_datapath = os.path.join(args.data_path, "video", "valid")
    val_labelpath = os.path.join(args.data_path, "label", "gt_valid.csv")
    val_dataset = TrimmedVideosDataset(
        val_datapath, val_labelpath, max_padding=args.max_padding
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        collate_fn=val_dataset.collate_fn,
    )

    # prepare model
    if args.model == "SCNN":
        model = SequentialCNN()

    # prepare optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # criterion
    criterion = nn.NLLLoss()

    # metric
    metric = MulticlassAccuracy()

    trainer = TrimmedTrainer(
        model,
        optimizer,
        criterion,
        train_dataloader,
        val_dataloader,
        writer,
        metric,
        args.save_dir,
        device,
        accumulate_gradient=args.accumulate_gradient,
    )

    trainer.fit(args.epochs)
