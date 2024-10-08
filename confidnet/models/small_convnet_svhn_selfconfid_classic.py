import torch.nn as nn
import torch.nn.functional as F

from confidnet.models.model import AbstractModel
from confidnet.models.small_convnet_svhn import Conv2dSame


class SmallConvNetSVHNSelfConfidClassic(AbstractModel):
    def __init__(self, config_args, device):
        super().__init__(config_args, device)
        self.feature_dim = config_args["model"]["feature_dim"]
        self.conv1 = Conv2dSame(config_args["data"]["input_channels"], 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = Conv2dSame(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.3)

        self.conv3 = Conv2dSame(32, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = Conv2dSame(64, 64, 3)
        self.bn4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.3)

        self.conv5 = Conv2dSame(64, 128, 3)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = Conv2dSame(128, 128, 3)
        self.bn6 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(2048, self.feature_dim)
        self.dropout4 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.feature_dim, config_args["data"]["num_classes"])

        self.uncertainty1 = nn.Linear(self.feature_dim, 400)
        self.uncertainty2 = nn.Linear(400, 400)
        self.uncertainty3 = nn.Linear(400, 400)
        self.uncertainty4 = nn.Linear(400, 400)
        self.uncertainty5 = nn.Linear(400, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.bn1(out)
        out = F.relu(self.conv2(out))
        out = self.bn2(out)
        out = self.maxpool1(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.3, training=self.training)
        else:
            out = self.dropout1(out)

        out = F.relu(self.conv3(out))
        out = self.bn3(out)
        out = F.relu(self.conv4(out))
        out = self.bn4(out)
        out = self.maxpool2(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.3, training=self.training)
        else:
            out = self.dropout2(out)

        out = F.relu(self.conv5(out))
        out = self.bn5(out)
        out = F.relu(self.conv6(out))
        out = self.bn6(out)
        out = self.maxpool3(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.3, training=self.training)
        else:
            out = self.dropout3(out)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        if self.mc_dropout:
            out = F.dropout(out, 0.3, training=self.training)
        else:
            out = self.dropout4(out)

        uncertainty = F.relu(self.uncertainty1(out))
        uncertainty = F.relu(self.uncertainty2(uncertainty))
        uncertainty = F.relu(self.uncertainty3(uncertainty))
        uncertainty = F.relu(self.uncertainty4(uncertainty))
        uncertainty = self.uncertainty5(uncertainty)
        pred = self.fc2(out)
        return pred, uncertainty
