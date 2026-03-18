
from typing import Iterable

import torch
import torch.nn as nn


__all__ = ['inception1d']


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def bn_drop_lin(n_in, n_out, bn=True, p=0., actn=None, layer_norm=False, permute=False):
    '''
    Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`. 
    permute for input of the form B,Seq,Feat"
    '''
    layers=[]
    if(permute):
        layers.append(LambdaLayer(lambda x: x.permute(0,2,1)))
    if(bn):
        if(layer_norm is False):
            layers.append(nn.BatchNorm1d(n_in))
        else:
            layers.append(nn.LayerNorm(n_in))
    if(permute):
        layers.append(LambdaLayer(lambda x: x.permute(0,2,1)))
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers


class Flatten(nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"
    def __init__(self, full:bool=False): 
        super().__init__()
        self.full = full
    def forward(self, x): return x.view(-1) if self.full else x.view(x.size(0), -1)


class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


def create_head1d(nf, nc, lin_ftrs=None, ps=0.5, bn:bool=True, act="relu", concat_pooling=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc] #was [nf, 512,nc]
    ps = [ps] if not isinstance(ps,Iterable) else ps
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True) if act=="relu" else nn.ELU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.AdaptiveAvgPool1d(1), Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
        layers += bn_drop_lin(ni,no,bn,p,actn)
    return nn.Sequential(*layers)


def conv(in_planes, out_planes, kernel_size=3, stride=1):
    "convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)

def noop(x): return x


class InceptionBlock1d(nn.Module):
    def __init__(self, ni, nb_filters, kss, stride=1, act='linear', bottleneck_size=32):
        super().__init__()
        self.bottleneck = conv(ni, bottleneck_size, 1, stride) if (bottleneck_size>0) else noop

        self.convs = nn.ModuleList([conv(bottleneck_size if (bottleneck_size>0) else ni, nb_filters, ks) for ks in kss])
        self.conv_bottle = nn.Sequential(nn.MaxPool1d(3, stride, padding=1), conv(ni, nb_filters, 1))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d((len(kss)+1)*nb_filters), nn.ReLU())

    def forward(self, x):
        #print("block in",x.size())
        bottled = self.bottleneck(x)
        out = self.bn_relu(torch.cat([c(bottled) for c in self.convs]+[self.conv_bottle(x)], dim=1))
        return out


class Shortcut1d(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.act_fn=nn.ReLU(True)
        self.conv=conv(ni, nf, 1)
        self.bn=nn.BatchNorm1d(nf)

    def forward(self, inp, out):
        #print("sk",out.size(), inp.size(), self.conv(inp).size(), self.bn(self.conv(inp)).size)
        #input()
        return self.act_fn(out + self.bn(self.conv(inp)))


class InceptionBackbone(nn.Module):
    def __init__(self, input_channels, kss, depth, bottleneck_size, nb_filters, use_residual):
        super().__init__()

        self.depth = depth
        assert((depth % 3) == 0)
        self.use_residual = use_residual

        n_ks = len(kss) + 1
        self.im = nn.ModuleList([InceptionBlock1d(input_channels if d==0 else n_ks*nb_filters,nb_filters=nb_filters,kss=kss, bottleneck_size=bottleneck_size) for d in range(depth)])
        self.sk = nn.ModuleList([Shortcut1d(input_channels if d==0 else n_ks*nb_filters, n_ks*nb_filters) for d in range(depth//3)])

    def forward(self, x):

        input_res = x
        for d in range(self.depth):
            x = self.im[d](x)
            if self.use_residual and d % 3 == 2:
                x = (self.sk[d//3])(input_res, x)
                input_res = x.clone()
        return x


class Inception1dOld(nn.Module):
    '''inception time architecture'''
    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        kss=[39,19,9],
        depth=6,
        bottleneck_size=32,
        nb_filters=32,
        use_residual=True,
        lin_ftrs_head=None,
        ps_head=0.5,
        bn_head=True,
        act_head="relu",
        concat_pooling=True,
    ):
        super().__init__()
        layers = [
            InceptionBackbone(
                input_channels=in_channels,
                kss=kss,
                depth=depth,
                bottleneck_size=bottleneck_size,
                nb_filters=nb_filters,
                use_residual=use_residual,
            )
        ]

        n_ks = len(kss) + 1
        #head
        head = create_head1d(
            n_ks * nb_filters,
            nc=num_classes,
            lin_ftrs=lin_ftrs_head,
            ps=ps_head,
            bn=bn_head,
            act=act_head,
            concat_pooling=concat_pooling,
        )
        layers.append(head)
        #layers.append(AdaptiveConcatPool1d())
        #layers.append(Flatten())
        #layers.append(nn.Linear(2*n_ks*nb_filters, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)

    def get_layer_groups(self):
        depth = self.layers[0].depth
        if(depth>3):
            return ((self.layers[0].im[3:],self.layers[0].sk[1:]),self.layers[-1])
        else:
            return (self.layers[-1])

    def get_output_layer(self):
        return self.layers[-1][-1]

    def set_output_layer(self,x):
        self.layers[-1][-1] = x


class Inception1d(Inception1dOld):
    '''inception time architecture'''
    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        kss=[39,19,9],
        depth=6,
        bottleneck_size=32,
        nb_filters=32,
        use_residual=True,
        ps_head=0.5,
        bn_head=True,
        concat_pooling=True,
    ):
        super(Inception1dOld, self).__init__()
        layers = [
            InceptionBackbone(
                input_channels=in_channels,
                kss=kss,
                depth=depth,
                bottleneck_size=bottleneck_size,
                nb_filters=nb_filters,
                use_residual=use_residual,
            )
        ]
        n_ks = len(kss) + 1  # 4

        self.layers = nn.Sequential(*layers)
        self.pool = AdaptiveConcatPool1d() if concat_pooling else nn.AdaptiveAvgPool1d(1)
        
        self.feature_dim = n_ks * nb_filters * (2 if concat_pooling else 1)

        self.bn = nn.BatchNorm1d(self.feature_dim) if bn_head else nn.Identity()
        self.dropout = nn.Dropout(ps_head) if ps_head > 0 else nn.Identity()

        self.reset_head(num_classes)

    def reset_head(self, num_classes):
        self.num_classes = num_classes
        if num_classes is not None:
            device = next(self.parameters()).device
            self.head = nn.Linear(self.feature_dim, num_classes)
            self.head.to(device)
        else:
            self.head = nn.Identity()

    def forward_feature(self, x):
        x = self.layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        return x

    def forward(self, x, return_feature=False):
        x = self.forward_feature(x)
        out = self.dropout(x)
        out = self.head(x)
        if return_feature:
            return out, x
        else:
            return out


def inception1d(in_channels, num_classes, old_version: bool = False, **kwargs):
    """Constructs an Inception model
    """
    if old_version:
        model = Inception1dOld(
            in_channels=in_channels,
            num_classes=num_classes,
            kss=kwargs.get('kss', [39, 19, 9]),
            depth=kwargs.get('depth', 6),
            bottleneck_size=kwargs.get('bottleneck_size', 32),
            nb_filters=kwargs.get('nb_filters', 32),
            use_residual=kwargs.get('use_residual', True),
            lin_ftrs_head=kwargs.get('lin_ftrs_head', None),
            ps_head=kwargs.get('ps_head', 0.5),
            bn_head=kwargs.get('bn_head', True),
            act_head=kwargs.get('act_head', "relu"),
            concat_pooling=kwargs.get('concat_pooling', True),
        )
    else:
        model = Inception1d(
            in_channels=in_channels,
            num_classes=num_classes,
            kss=kwargs.get('kss', [39, 19, 9]),
            depth=kwargs.get('depth', 6),
            bottleneck_size=kwargs.get('bottleneck_size', 32),
            nb_filters=kwargs.get('nb_filters', 32),
            use_residual=kwargs.get('use_residual', True),
            ps_head=kwargs.get('ps_head', 0.5),
            bn_head=kwargs.get('bn_head', True),
            concat_pooling=kwargs.get('concat_pooling', True),
        )

    return model
