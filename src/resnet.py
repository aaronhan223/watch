import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    A basic 'ResNet v1' block for CIFAR:
    Conv -> BN -> ReLU -> Conv -> BN, then add the shortcut (post-activation).
    """
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        
        self.shortcut = nn.Sequential()
        # If dimension or stride changes, adjust the shortcut
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, 
                                      stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut
        shortcut = self.shortcut(x)
        out += shortcut
        
        out = F.relu(out)  # post-activation
        return out

class ResNetCIFARv1(nn.Module):
    """
    A 'v1' ResNet for CIFAR-10 using BasicBlock (2-layer blocks).
    By default, we build 'ResNet-20' with 3 stages of blocks: [3,3,3].
    
    If you want ResNet-32, set num_blocks_per_stage=5 => [5,5,5].
    
    The network is:
    - initial conv (3 -> 16)
    - stage1: out_planes=16, stride=1
    - stage2: out_planes=32, stride=2
    - stage3: out_planes=64, stride=2
    - final linear -> 10 (CIFAR-10 classes)
    """
    def __init__(self, block, num_blocks_per_stage=[3,3,3], num_classes=10):
        super().__init__()
        self.in_planes = 16  # initial number of channels

        # Initial 3x3 conv
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Build stages
        self.stage1 = self._make_layer(block, out_planes=16,  num_blocks=num_blocks_per_stage[0], stride=1)
        self.stage2 = self._make_layer(block, out_planes=32,  num_blocks=num_blocks_per_stage[1], stride=2)
        self.stage3 = self._make_layer(block, out_planes=64,  num_blocks=num_blocks_per_stage[2], stride=2)

        # Pool + linear
        self.linear = nn.Linear(64 * block.expansion, num_classes)

        # (Optional) weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_planes, num_blocks, stride):
        """
        Create a stage containing `num_blocks` blocks in a row, 
        possibly with downsampling stride for the first block.
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, out_planes, s))
            self.in_planes = out_planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # initial conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # stages
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)

        # global average pool
        out = F.avg_pool2d(out, out.size(3))  # or out.size(2) as H = W
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet20(num_classes=10):
    """
    Classic ResNet-20 for CIFAR (depth=20) => 3 stages each with 3 BasicBlocks.
    """
    return ResNetCIFARv1(BasicBlock, [3,3,3], num_classes=num_classes)

def ResNet32(num_classes=10):
    """
    Classic ResNet-32 for CIFAR (depth=32) => 3 stages each with 5 BasicBlocks.
    """
    return ResNetCIFARv1(BasicBlock, [5,5,5], num_classes=num_classes)