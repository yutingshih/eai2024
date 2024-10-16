import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # defines two consecutive convolutional layers within the residual block
        self.left = nn.Sequential(
            nn.Conv2d(
                inchannel,
                outchannel,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut, in order to maintain consistency with the structure of the two convolutional layers' results, some processing is required here.
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    inchannel, outchannel, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(outchannel),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.left(x)
        # Adding the outputs of the two convolutional layers to the processed x achieves the fundamental structure of ResNet
        out = out + self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3, self.inchannel, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # make layers
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, num_classes)

    # This function is primarily used to repeat the same residual block
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (
            num_blocks - 1
        )  # first block has `stride`, others are 1
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet18(num_classes=10) -> nn.Module:
    return ResNet(ResBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes=10) -> nn.Module:
    return ResNet(ResBlock, [3, 4, 6, 3], num_classes)


if __name__ == "__main__":
    print(ResNet18())
    print(ResNet34())
