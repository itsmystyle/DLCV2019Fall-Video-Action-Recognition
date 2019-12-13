import os
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.utils import set_random_seed
from modules.dataset import TrimmedVideosDataset
from modules.models.SCNN import SequentialCNN
from modules.models.RCNN import RecurrentCNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sequential trimmed video.")
    parser.add_argument(
        "model", type=str, default="SCNN", help="Which model to train (SCNN or RCNN)."
    )
    parser.add_argument("video_path", type=str, help="Path to video directory.")
    parser.add_argument(
        "video_label_path", type=str, help="Path to video label directory."
    )
    parser.add_argument("model_dir", type=str, help="Where to load trained model.")
    parser.add_argument("output_dir", type=str, help="Output directory.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--max_padding", type=int, default=30, help="Max padding length of frames."
    )
    parser.add_argument("--ds_factor", type=int, default=12, help="Down-sample factor.")
    parser.add_argument("--rescale_factor", type=int, default=1, help="Rescale factor.")
    parser.add_argument(
        "--sorting",
        action="store_true",
        help="Whether to sort by video length per batch.",
    )
    parser.add_argument(
        "--n_workers", type=int, default=8, help="Number of worker for dataloader."
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    set_random_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare dataset
    dataset = TrimmedVideosDataset(
        args.video_path,
        args.video_label_path,
        max_padding=args.max_padding,
        rescale_factor=(1.0 / args.rescale_factor),
        downsample_factor=args.ds_factor,
        sorting=args.sorting,
        test=True,
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        collate_fn=dataset.collate_fn,
    )

    # prepare model
    if args.model == "SCNN":
        model = SequentialCNN()
        output_dir = os.path.join(args.output_dir, "p1_valid.txt")
    elif args.model == "RCNN":
        model = RecurrentCNN()
        output_dir = os.path.join(args.output_dir, "p2_result.txt")

    model.load_state_dict(torch.load(args.model_dir))
    model.to(device)
    model.eval()

    preds_ls = []

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            frames = batch["frames"].to(device)
            frames_len = batch["frames_len"]
            sorted_idx = batch["sorted_idx"]

            preds = model(frames, frames_len)
            preds = torch.exp(preds).max(dim=1)[1].detach().cpu().numpy()

            preds_ls.append(preds[sorted_idx])

    preds_ls = np.concatenate(preds_ls)

    with open(output_dir, "w") as fout:
        for pred in preds_ls:
            fout.write(str(pred) + "\n")
