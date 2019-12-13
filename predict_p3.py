import os
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.utils import set_random_seed
from modules.dataset import FullLengthVideosDataset
from modules.models.LRCNN import SeqRecurrentCNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict full-length videos.")
    parser.add_argument("video_path", type=str, help="Path to video directory.")
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
    dataset = FullLengthVideosDataset(
        args.video_path, None, length=100, overlap=0, sorting=True, test=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # prepare model
    model = SeqRecurrentCNN()

    model.load_state_dict(torch.load(args.model_dir))
    model.to(device)
    model.eval()

    preds_ls = {}

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            frames = batch["frames"].to(device)
            frames_len = batch["frames_len"]
            sorted_idx = batch["sorted_idx"]
            category = batch["category"]

            preds = model(frames, frames_len)
            preds = torch.exp(preds).max(dim=2)[1].detach().cpu().numpy()

            for _idx in sorted_idx:
                cat = category[_idx][0]
                if cat not in preds_ls:
                    preds_ls[cat] = []
                preds_ls[cat] += preds[_idx][: frames_len[_idx]].tolist()

    for k, v in preds_ls.items():
        output_dir = os.path.join(args.output_dir, "{}.txt".format(k))
        with open(output_dir, "w") as fout:
            for label in v:
                fout.write(str(label) + "\n")
