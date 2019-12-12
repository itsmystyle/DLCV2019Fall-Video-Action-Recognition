import torch.nn as nn
import torchvision.models as models


class SequentialCNN(nn.Module):
    def __init__(self):
        super(SequentialCNN, self).__init__()

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
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
        )

        self.classifier = nn.Sequential(nn.Linear(64, 11), nn.LogSoftmax(dim=1))

    def forward(self, frames, frames_len):
        bs, ts, c, w, h = frames.shape
        frames = frames.view(-1, c, w, h)
        frames = self.backbone(frames)
        frames = frames.view(bs * ts, -1)
        frames = self.linears(frames)
        frames = frames.view(bs, ts, -1)
        frames = frames.mean(dim=1)
        frames = self.classifier(frames)

        return frames

    def extract_features(self, frames):
        pass
