import os
import glob
import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
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


class FullLengthVideosDataset(Dataset):
    def __init__(
        self,
        video_path,
        video_label_path,
        rescale_factor=1,
        length=100,
        overlap=15,
        test=False,
        sorting=False,
        transform=None,
    ):
        self.video_path = video_path
        self.video_label_path = video_label_path
        self.rescale_factor = 1.0 / rescale_factor
        self.length = length
        self.overlap = overlap
        self.test = test
        self.sorting = sorting

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
            )

        print("preparing data...")
        self.categories = sorted(os.listdir(self.video_path))
        self.datas = []
        for category in tqdm(self.categories, total=len(self.categories)):

            # frames
            frames_path = sorted(glob.glob(os.path.join(self.video_path, category, "*")))
            frames = []
            for path in frames_path:
                # frame = skimage.io.imread(path)
                # frame = skimage.transform.rescale(
                #     frame,
                #     rescale_factor,
                #     mode="constant",
                #     preserve_range=True,
                #     multichannel=True,
                #     anti_aliasing=True,
                # ).astype(np.uint8)
                # frame.append(frame)
                frame = Image.open(path)
                frames.append(np.array(frame).astype(np.uint8))
            frames = np.stack(frames)

            # labels
            if not self.test:
                labels_path = os.path.join(self.video_label_path, "{}.txt".format(category))
                with open(labels_path, "r") as fin:
                    labels = fin.readlines()
                labels = np.array([int(i.strip()) for i in labels])
            else:
                labels = np.array([0] * len(frames_path))

            self.datas += self.trim_frames(frames, labels)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index]

    def collate_fn(self, datas):
        batch = {}

        lens = [data[0].shape[0] for data in datas]
        padding_len = max(lens)

        if self.sorting:
            # sort whole datas with its video length
            sorted_idx = np.argsort(lens)[::-1]
            datas = [datas[idx] for idx in sorted_idx]

        # frames_len
        frames_len = [data[0].shape[0] for data in datas]
        batch["frames_len"] = frames_len

        # frames
        batch_size = len(datas)
        width, height, channel = datas[0][0].shape[1:]
        frames = torch.zeros((batch_size, padding_len, channel, width, height))
        frames[:, :, 0, :, :] = (frames[:, :, 0, :, :] - MEAN[0]) / STD[0]
        frames[:, :, 1, :, :] = (frames[:, :, 1, :, :] - MEAN[1]) / STD[1]
        frames[:, :, 2, :, :] = (frames[:, :, 2, :, :] - MEAN[2]) / STD[2]
        for idx, (data, _) in enumerate(datas):
            for step, frame in enumerate(data):
                frames[idx, step] = self.transform(frame)
        batch["frames"] = frames.float()

        if not self.test:
            # labels
            labels = np.zeros((batch_size, padding_len), dtype=np.int64)
            for idx, (_, data) in enumerate(datas):
                labels[idx, : data.shape[0]] = data
            batch["labels"] = torch.tensor(labels).long()

        return batch

    def trim_frames(self, frames, labels):
        chunk_size = frames.shape[0] // (self.length - self.overlap)

        frame_chunks = np.array_split(frames, chunk_size)
        label_chunks = np.array_split(labels, chunk_size)

        final_chunks = []

        for i in range(chunk_size):
            if self.overlap > 0:
                if i == 0:
                    final_chunks.append((frame_chunks[i], label_chunks[i]))
                else:
                    frame_chunk = np.concatenate(
                        (frame_chunks[i - 1][-self.overlap:], frame_chunks[i])
                    )
                    label_chunk = np.concatenate(
                        (label_chunks[i - 1][-self.overlap:], label_chunks[i])
                    )
                    final_chunks.append((frame_chunk, label_chunk))
            else:
                final_chunks.append((frame_chunks[i], label_chunks[i]))

        return final_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing Videos dataset.")
    parser.add_argument(
        "--mode", type=str, default="trimmed", help="Which dataset to test. (trimmed or full)",
    )
    parser.add_argument("video_path", type=str, help="Path to video directory.")
    parser.add_argument("video_label_path", type=str, help="Path to video label directory.")
    parser.add_argument("--ds_factor", type=int, default=12, help="Down-sample factor.")
    parser.add_argument("--rescale_factor", type=int, default=1, help="Rescale factor.")
    parser.add_argument(
        "--sorting", action="store_true", help="Whether to sort by video length per batch.",
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
        dataset = FullLengthVideosDataset(
            args.video_path,
            args.video_label_path,
            downsample_factor=args.ds_factor,
            rescale_factor=args.rescale_factor,
            sorting=args.sorting,
        )

    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn,
    )

    for batch in tqdm(dataloader, total=len(dataloader)):
        frames_len = batch["frames_len"].to("cuda")
        labels = batch["labels"].to("cuda")
        frames = batch["frames"].to("cuda")
