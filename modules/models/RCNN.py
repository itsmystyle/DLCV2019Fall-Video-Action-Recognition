import torch.nn as nn
import torchvision.models as models


class RecurrentCNN(nn.Module):
    def __init__(self):
        super(RecurrentCNN, self).__init__()

        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))

        # Freeze backbone
        for child in self.backbone.children():
            for param in child.parameters():
                param.requires_grad = False

        self.conv = nn.Sequential(
            nn.Conv2d(2048, 2048, 3),
            nn.BatchNorm2d(2048),
            nn.MaxPool2d(3, 2),
            nn.ReLU(True),
        )

        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.rnn = nn.LSTM(2048 * 2 * 3, 2048, bidirectional=True, batch_first=True)

        self.classifier = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Linear(64, 11),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, frames, frames_len):
        bs, ts, c, w, h = frames.shape
        frames = frames.view(-1, c, w, h)
        frames = self.backbone(frames)
        frames = self.conv(frames)
        frames = frames.view(bs, ts, -1)
        frames = nn.utils.rnn.pack_padded_sequence(frames, frames_len, batch_first=True)
        frames, (hn, cn) = self.rnn(frames)
        # frames, _ = nn.utils.rnn.pad_packed_sequence(frames, batch_first=True)
        frames = hn[0] + hn[1]
        frames = self.classifier(frames)

        return frames

    def extract_features(self, frames):
        pass
