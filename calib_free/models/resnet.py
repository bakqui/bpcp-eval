"""
resnet for 1-d signal data, pytorch version
 
Originally taken from Shenda Hong, Oct 2019
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['pulse_ppg', 'papagei_s']


class StandardScaler1d(nn.Module):
    """
    feature-wise standard scaler for 1D  data
    """
    def __init__(
        self,
        feature_dim: int = 1,
        feature_mean: Optional[torch.Tensor] = None,
        feature_std: Optional[torch.Tensor] = None,
    ):
        super(StandardScaler1d, self).__init__()
        if feature_mean is None:
            feature_mean = torch.zeros(feature_dim)
        if feature_std is None:
            feature_std = torch.ones(feature_dim)
        self.feature_mean = nn.Parameter(feature_mean, requires_grad=False)
        self.feature_std = nn.Parameter(feature_std, requires_grad=False)

    def forward(self, x):
        # x: (n_samples, n_channel, n_length)
        out = (x - self.feature_mean) / self.feature_std

        return out


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, 
                 is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out


class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks, each block has 2 convs
        n_classes: number of classes
        
    """

    def __init__(
        self,
        in_channels,
        num_classes=None,
        base_filters=128,
        kernel_size=11, 
        stride=2,
        groups=1,
        n_block=12,
        finalpool=None,
        downsample_gap=2,
        increasefilter_gap=4,
        use_bn=True,
        use_do=True,
        verbose=False,
    ):
        super().__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, 
                                                out_channels=base_filters, 
                                                kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # channels and out_channels
            if is_first_block:
                channels = base_filters
                out_channels = channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                # when i_block=11, in_channels=512
                channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = channels * 2
                else:
                    out_channels = channels
            
            tmp_block = BasicBlock(
                in_channels=channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)
            
        self.instnorm = nn.InstanceNorm1d(in_channels)
        self.finalpool = finalpool
        self.feature_dim = out_channels

        self.reset_standard_scaler()
        self.reset_head(num_classes)

    def reset_standard_scaler(
        self,
        feature_mean: Optional[torch.Tensor] = None,
        feature_std: Optional[torch.Tensor] = None,
    ):
        # add feature normalization after backbone (no training here)
        if feature_mean is None and feature_std is None:
            self.standard_scaler = nn.Identity()
        else:
            device = next(self.parameters()).device
            self.standard_scaler = StandardScaler1d(
                feature_dim=self.feature_dim,
                feature_mean=feature_mean,
                feature_std=feature_std,
            )
            self.standard_scaler.to(device)

    def reset_head(self, num_classes):
        self.num_classes = num_classes
        if num_classes is not None:
            device = next(self.parameters()).device
            self.head = nn.Linear(self.feature_dim, num_classes)
            self.head.to(device)
        else:
            self.head = nn.Identity()

    def forward_feature(self, x):

        out = x
        
        out = self.instnorm(out)
        
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        if self.finalpool:
            if self.finalpool == "avg":
                out = torch.mean(out, dim=-1)
            elif self.finalpool == "max":
                out = torch.max(out, dim=-1)[0]
            else:
                print("invalid pool")
                import sys; sys.exit()
        
        return out

    def forward(self, x, return_feature=False):
        x = self.forward_feature(x)
        x = self.standard_scaler(x)
        out = self.head(x)
        if return_feature:
            return out, x
        else:
            return out


class ResNet1DMoE(nn.Module):
    """
    ResNet1D with Two Mixture of Experts (MoE) Regression Heads

    Parameters:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larger to 1 as ResNeXt
        n_block: number of blocks
        embedding_dim: dimension of the embedding output
        n_experts: number of expert models in the MoE head
    """

    def __init__(
        self,
        in_channels,
        num_classes=None,
        base_filters=32,
        kernel_size=3,
        stride=2,
        groups=1,
        n_block=18,
        embedding_dim=512,
        n_experts=3,
        downsample_gap=2,
        increasefilter_gap=4,
        use_bn=True,
        use_do=True,
        verbose=False,
        use_projection=False,
    ):
        super(ResNet1DMoE, self).__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.use_projection = use_projection
        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model
        self.n_experts = n_experts

        # First block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # Residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            is_first_block = (i_block == 0)
            downsample = (i_block % self.downsample_gap == 1)
            
            in_channels = base_filters if is_first_block else int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
            out_channels = in_channels * 2 if (i_block % self.increasefilter_gap == 0 and i_block != 0) else in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                groups=self.groups, 
                downsample=downsample, 
                use_bn=self.use_bn, 
                use_do=self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        if self.use_projection:
            self.projector = nn.Sequential(
                nn.Linear(out_channels, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        # Final layers
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_channels, embedding_dim)
        # Mixture of Experts (MoE) Head 1 (mt_regression 1)
        self.expert_layers_1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_channels, out_channels // 2),
                nn.ReLU(),
                nn.Linear(out_channels // 2, 1)
            ) for _ in range(self.n_experts)
        ])
        self.gating_network_1 = nn.Sequential(
            nn.Linear(out_channels, self.n_experts),
            nn.Softmax(dim=1)  # Softmax to produce weights for the experts
        )

        # Mixture of Experts (MoE) Head 2 (mt_regression 2)
        self.expert_layers_2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_channels, out_channels // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(out_channels // 2, 1)
            ) for _ in range(self.n_experts)
        ])
        self.gating_network_2 = nn.Sequential(
            nn.Linear(out_channels, self.n_experts),
            nn.Softmax(dim=1)  # Softmax to produce weights for the experts
        )

        self.feature_dim = 128 if self.use_projection else embedding_dim

        self.reset_standard_scaler()
        self.reset_head(num_classes)

    def reset_standard_scaler(
        self,
        feature_mean: Optional[torch.Tensor] = None,
        feature_std: Optional[torch.Tensor] = None,
    ):
        # add feature normalization after backbone (no training here)
        if feature_mean is None and feature_std is None:
            self.standard_scaler = nn.Identity()
        else:
            device = next(self.parameters()).device
            self.standard_scaler = StandardScaler1d(
                feature_dim=self.feature_dim,
                feature_mean=feature_mean,
                feature_std=feature_std,
            )
            self.standard_scaler.to(device)

    def reset_head(self, num_classes):
        if num_classes is not None:
            device = next(self.parameters()).device
            self.head = nn.Linear(self.feature_dim, num_classes)
            self.head.to(device)
        else:
            self.head = nn.Identity()

    def forward_feature(self, x):
        out = x
        
        # First conv layer
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # Residual blocks
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # Final layers
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)

        if self.use_projection:
            embed = self.projector(out)
        else:
            embed = self.dense(out)

        return out, embed

    def forward(self, x, return_feature=False, return_moe=False):
        x, embed = self.forward_feature(x)
        embed = self.standard_scaler(embed)
        out = self.head(embed)

        if return_moe:
            expert_outputs_1 = torch.stack([expert(x) for expert in self.expert_layers_1], dim=1)  # (batch_size, n_experts, 1)
            gate_weights_1 = self.gating_network_1(x)  # (batch_size, n_experts)
            out_moe1 = torch.sum(gate_weights_1.unsqueeze(2) * expert_outputs_1, dim=1)  # Weighted sum of experts
            expert_outputs_2 = torch.stack([expert(x) for expert in self.expert_layers_2], dim=1)  # (batch_size, n_experts, 1)
            gate_weights_2 = self.gating_network_2(x)  # (batch_size, n_experts)
            out_moe2 = torch.sum(gate_weights_2.unsqueeze(2) * expert_outputs_2, dim=1)  # Weighted sum of experts
            return out, out_moe1, out_moe2, embed
        elif return_feature:
            return out, embed
        else:
            return out


def pulse_ppg(in_channels=1, num_classes=None, finalpool="max", **kwargs):
    """PulsePPG default 1D ResNet model
    """
    model = ResNet1D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=128,
        kernel_size=11,
        stride=2,
        groups=1,
        n_block=12,
        finalpool=finalpool,
        downsample_gap=2,
        increasefilter_gap=4,
        use_bn=True,
        use_do=kwargs.get("use_do", True),
        verbose=kwargs.get("verbose", False),
    )

    return model


def papagei_s(in_channels=1, num_classes=None, **kwargs):
    """PaPaGei-S: PaPaGei default 1D ResNet model
    """
    model = ResNet1DMoE(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=32,
        kernel_size=3,
        stride=2,
        groups=1,
        n_block=18,
        embedding_dim=512,
        n_experts=3,
        downsample_gap=2,
        increasefilter_gap=4,
        use_bn=True,
        use_projection=False,
        use_do=kwargs.get("use_do", True),
        verbose=kwargs.get("verbose", False),
    )

    return model
