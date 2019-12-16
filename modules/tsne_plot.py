import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from dataset import TrimmedVideosDataset
from models.SCNN import SequentialCNN
from models.RCNN import RecurrentCNN
from utils import set_random_seed

set_random_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

video_path = "/tmp2/itsmystyle/h4_data/TrimmedVideos/video/valid/"
video_label_path = "/tmp2/itsmystyle/h4_data/TrimmedVideos/label/gt_valid.csv"

# video_path = '/tmp2/itsmystyle/h4_data/TrimmedVideos/video/train/'
# video_label_path = '/tmp2/itsmystyle/h4_data/TrimmedVideos/label/gt_train.csv'

dataset = TrimmedVideosDataset(
    video_path,
    video_label_path,
    max_padding=30,
    rescale_factor=(1.0 / 1),
    downsample_factor=12,
    sorting=True,
)
dataloader = DataLoader(
    dataset, shuffle=False, batch_size=4, num_workers=8, collate_fn=dataset.collate_fn
)

model = SequentialCNN()
model.load_state_dict(torch.load("../models/SCNN/model_best_0.42783_1.97644.pth.tar"))
model.to(device)
model.eval()

# model = RecurrentCNN()
# model.load_state_dict(torch.load('../models/RCNN/model_best_0.49675_1.54730.pth.tar'))
# model.to(device)
# model.eval()

preds_ls = []
labels_ls = []

with torch.no_grad():
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        frames = batch["frames"].to(device)
        frames_len = batch["frames_len"]
        sorted_idx = batch["sorted_idx"]
        labels = batch["labels"]

        preds = model.extract_features(frames)
        #         preds = model.extract_features(frames, frames_len)

        preds_ls.append(preds.cpu().numpy())
        labels_ls.append(labels.cpu().numpy())

preds_ls = np.concatenate(preds_ls)
labels_ls = np.concatenate(labels_ls)

tsne = TSNE(n_components=2, verbose=1)

X_tsne = tsne.fit_transform(preds_ls)


plt.figure(figsize=(8, 6))
for i in range(11):
    select_idxs = np.where(labels_ls == i)[0]
    plt.scatter(
        x=X_tsne[select_idxs, 0],
        y=X_tsne[select_idxs, 1],
        c="C{}".format(i) if i != 10 else "black",
        label=i,
    )
plt.legend()
plt.savefig("tsne.png")
plt.close()
