import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from modules.reader import readShortVideo, getVideoList


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class TrimmedVideosDataset(Dataset):
    def __init__(
        self,
        video_path,
        video_label_path,
        downsample_factor=12,
        rescale_factor=1,
        max_padding=24,
        test=False,
        sorting=False,
    ):
        self.video_path = video_path
        self.video_list = getVideoList(video_label_path)
        self.len = len(self.video_list["Action_labels"])
        self.downsample_factor = downsample_factor
        self.rescale_factor = 1.0 / rescale_factor
        self.max_padding = max_padding
        self.test = test
        self.sorting = sorting

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        )

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        video = readShortVideo(
            self.video_path,
            self.video_list["Video_category"][index],
            self.video_list["Video_name"][index],
            rescale_factor=self.rescale_factor,
            downsample_factor=self.downsample_factor,
        )

        if not self.test:
            label = int(self.video_list["Action_labels"][index])
        else:
            label = None

        return video, int(label)

    def collate_fn(self, datas):
        batch = {}

        if self.sorting:
            # sort whole datas with its video length
            frames_len = np.array([data[0].shape[0] for data in datas])
            sorted_idx = np.argsort(frames_len)[::-1]
            # datas = np.array(datas)[sorted_idx]
            datas = [datas[idx] for idx in sorted_idx]

        # frames_len
        frames_len = [min(data[0].shape[0], self.max_padding) for data in datas]
        batch["frames_len"] = frames_len

        # frames
        batch_size = len(datas)
        max_padding_len = min(max(frames_len), self.max_padding)
        width, height, channel = datas[0][0].shape[1:]
        frames = torch.zeros((batch_size, max_padding_len, channel, width, height))
        for idx, (data, _) in enumerate(datas):
            for step, frame in enumerate(data[:max_padding_len]):
                frames[idx, step] = self.transform(frame)
        batch["frames"] = frames.float()

        if not self.test:
            # labels
            labels = [data[1] for data in datas]
            batch["labels"] = torch.tensor(labels).long()

        return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing Videos dataset.")
    parser.add_argument(
        "--mode",
        type=str,
        default="trimmed",
        help="Which dataset to test. (trimmed or full)",
    )
    parser.add_argument("video_path", type=str, help="Path to video directory.")
    parser.add_argument(
        "video_label_path", type=str, help="Path to video label directory."
    )
    parser.add_argument("--ds_factor", type=int, default=12, help="Down-sample factor.")
    parser.add_argument("--rescale_factor", type=int, default=1, help="Rescale factor.")
    parser.add_argument(
        "--sorting",
        action="store_true",
        help="Whether to sort by video length per batch.",
    )

    args = parser.parse_args()

    if args.mode == "trimmed":
        dataset = TrimmedVideosDataset(
            args.video_path,
            args.video_label_path,
            downsample_factor=args.ds_factor,
            rescale_factor=args.rescale_factor,
            sorting=args.sorting,
        )
    else:
        pass

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    for batch in tqdm(dataloader, total=len(dataloader)):
        frames_len = batch["frames_len"].to("cuda")
        labels = batch["labels"].to("cuda")
        frames = batch["frames"].to("cuda")
