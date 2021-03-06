import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SeqRecurrentCNN(nn.Module):
    def __init__(self):
        super(SeqRecurrentCNN, self).__init__()

        self.hidden_dim = 1024

        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))

        # Freeze backbone
        for child in self.backbone.children():
            for param in child.parameters():
                param.requires_grad = False

        self.linears = nn.Sequential(
            nn.Linear(2048 * 2 * 4, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(True),
        )

        self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 11),
            nn.LogSoftmax(dim=2),
        )

    def forward(self, frames, frames_len):
        bs, ts, c, w, h = frames.shape
        frames = frames.view(-1, c, w, h)
        frames = self.backbone(frames)
        frames = frames.view(bs * ts, -1)
        frames = self.linears(frames)
        frames = frames.view(bs, ts, -1)
        frames = nn.utils.rnn.pack_padded_sequence(frames, frames_len, batch_first=True)
        frames, (hn, cn) = self.rnn(frames)
        frames, _ = nn.utils.rnn.pad_packed_sequence(frames, batch_first=True)
        frames = self.classifier(F.relu(frames))

        return frames

    def extract_features(self, frames, frames_len):
        bs, ts, c, w, h = frames.shape
        frames = frames.view(-1, c, w, h)
        frames = self.backbone(frames)
        frames = frames.view(bs * ts, -1)
        frames = self.linears(frames)
        frames = frames.view(bs, ts, -1)
        frames = nn.utils.rnn.pack_padded_sequence(frames, frames_len, batch_first=True)
        frames, (hn, cn) = self.rnn(frames)
        frames, _ = nn.utils.rnn.pad_packed_sequence(frames, batch_first=True)

        return F.relu(frames)
