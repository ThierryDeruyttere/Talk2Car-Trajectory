import torch.nn as nn
from torch.nn import init


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=None):
    if padding is None:
        padding_inside = (kernel_size - 1) // 2
    else:
        padding_inside = padding
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding_inside,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding_inside,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )


class FlowNetS(nn.Module):
    def __init__(
        self, input_channels=6, batchNorm=True, input_width=320, input_height=576
    ):
        super(FlowNetS, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, input_channels, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.conv7 = conv(
            self.batchNorm, 1024, 1024, kernel_size=1, stride=1, padding=0
        )
        self.conv8 = conv(
            self.batchNorm, 1024, 1024, kernel_size=1, stride=1, padding=0
        )

        if input_width == 600 and input_height == 400:
            fc1_input_features = 71680
        elif input_width == 300 and input_height == 200:
            fc1_input_features = 20480
        elif input_width == 150 and input_height == 100:
            fc1_input_features = 6144

        self.fc1 = nn.Linear(
            in_features=fc1_input_features, out_features=1024, bias=True
        )

        self.fc2 = nn.Linear(in_features=1024, out_features=1024, bias=True)

        # self.predict_6 = predict_flow(1024)
        # self.fc1 = nn.Linear(in_features=90, out_features=512, bias=True)
        # self.fc2 = nn.Linear(in_features=512, out_features=1024, bias=True)
        # self.fc3 = nn.Linear(in_features=1024, out_features=2048, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        out_conv8 = self.conv8(self.conv7(out_conv6))  # SDD
        out_fc1 = nn.functional.relu(
            self.fc1(out_conv8.view(out_conv6.size(0), -1))
        )  # SDD

        #         out_fc1 = nn.functional.relu(self.fc1(out_conv6.view(out_conv6.size(0), -1))) # CPI

        # predict = self.predict_6(out_conv8)
        # out_fc1 = nn.functional.relu(self.fc1(predict.view(predict.size(0), -1)))
        # out_fc2 = nn.functional.relu(self.fc2(out_fc1))
        # out_fc3 = nn.functional.relu(self.fc3(out_fc2))
        # out_conv7 = self.conv7(out_conv6)
        # out_conv8 = self.conv8(out_conv7)
        # out_fc2 = out_conv6.view(out_conv6.size(0), -1)
        # out_fc2 = self.fc1(out_conv8.view(out_conv8.size(0), -1))

        out_fc2 = nn.functional.relu(self.fc2(out_fc1))
        # out_fc2 = self.predict_6(out_conv6)
        return out_fc2