import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_, DropPath
import os
import sys
import torch.fft
import math
from torch.autograd import Variable
import numpy as np

import traceback

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


if 'DWCONV_IMPL' in os.environ:
    try:
        sys.path.append(os.environ['DWCONV_IMPL'])
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM


        def get_dwconv(dim, kernel, bias):
            return DepthWiseConv2dImplicitGEMM(dim, kernel, bias)
        # print('Using Megvii large kernel dw conv impl')
    except:
        print(traceback.format_exc())


        def get_dwconv(dim, kernel, bias):
            return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

        # print('[fail to use Megvii Large kernel] Using PyTorch large kernel dw conv impl')
else:
    def get_dwconv(dim, kernel, bias):
        return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

    # print('Using PyTorch large kernel dw conv impl')


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class HFConv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )
        if self.order == 3:
            self.conv0 = nn.Conv2d(self.dims[0], self.dims[1], 1)
            self.conv1 = nn.Conv2d(self.dims[1], self.dims[2], 1)
        else:
            if self.order == 4:
                self.conv0 = nn.Conv2d(self.dims[0], self.dims[1], 1)
                self.conv1 = nn.Conv2d(self.dims[1], self.dims[2], 1)
                self.conv2 = nn.Conv2d(self.dims[2], self.dims[3], 1)
            else:
                if self.order == 5:
                    self.conv0 = nn.Conv2d(self.dims[0], self.dims[1], 1)
                    self.conv1 = nn.Conv2d(self.dims[1], self.dims[2], 1)
                    self.conv2 = nn.Conv2d(self.dims[2], self.dims[3], 1)
                    self.conv3 = nn.Conv2d(self.dims[3], self.dims[4], 1)
                else:
                    if self.order == 2:
                        self.conv0 = nn.Conv2d(self.dims[0], self.dims[1], 1)



        self.focol = nn.ModuleList([
            BasicLayer(
                dim=self.dims[i + 1],
                depth=2,
                mlp_ratio=4.,
                drop=0.,
                drop_path=0.4,
                norm_layer=nn.LayerNorm,
                focal_window=9,
                focal_level=2,
                use_layerscale=False,
                use_checkpoint=False) for i in range(order - 1)]
        )
        if self.order == 3:
            self.focol0 = BasicLayer(dim=self.dims[0], depth=2, mlp_ratio=4., drop=0., drop_path=0.4,
                                     norm_layer=nn.LayerNorm, focal_window=9, focal_level=2, use_layerscale=False,
                                     use_checkpoint=False)
            self.focol1 = BasicLayer(dim=self.dims[1], depth=2, mlp_ratio=4., drop=0., drop_path=0.4,
                                     norm_layer=nn.LayerNorm, focal_window=9, focal_level=2, use_layerscale=False,
                                     use_checkpoint=False)
            self.focol2 = BasicLayer(
                dim=self.dims[2],
                depth=2,
                mlp_ratio=4.,
                drop=0.,
                drop_path=0.4,
                norm_layer=nn.LayerNorm,
                focal_window=9,
                focal_level=2,
                use_layerscale=False,
                use_checkpoint=False)
        else:
            if self.order == 4:
                self.focol0 = BasicLayer(
                    dim=self.dims[0],
                    depth=2,
                    mlp_ratio=4.,
                    drop=0.,
                    drop_path=0.4,
                    norm_layer=nn.LayerNorm,
                    focal_window=9,
                    focal_level=2,
                    use_layerscale=False,
                    use_checkpoint=False)

                self.focol1 = BasicLayer(
                    dim=self.dims[1],
                    depth=2,
                    mlp_ratio=4.,
                    drop=0.,
                    drop_path=0.4,
                    norm_layer=nn.LayerNorm,
                    focal_window=9,
                    focal_level=2,
                    use_layerscale=False,
                    use_checkpoint=False)

                self.focol2 = BasicLayer(
                    dim=self.dims[2],
                    depth=2,
                    mlp_ratio=4.,
                    drop=0.,
                    drop_path=0.4,
                    norm_layer=nn.LayerNorm,
                    focal_window=9,
                    focal_level=2,
                    use_layerscale=False,
                    use_checkpoint=False)

                self.focol3 = BasicLayer(
                    dim=self.dims[3],
                    depth=2,
                    mlp_ratio=4.,
                    drop=0.,
                    drop_path=0.4,
                    norm_layer=nn.LayerNorm,
                    focal_window=9,
                    focal_level=2,
                    use_layerscale=False,
                    use_checkpoint=False)
            else:
                if self.order == 5:
                    self.focol0 = BasicLayer(
                        dim=self.dims[0],
                        depth=2,
                        mlp_ratio=4.,
                        drop=0.,
                        drop_path=0.4,
                        norm_layer=nn.LayerNorm,
                        focal_window=9,
                        focal_level=2,
                        use_layerscale=False,
                        use_checkpoint=False)

                    self.focol1 = BasicLayer(
                        dim=self.dims[1],
                        depth=2,
                        mlp_ratio=4.,
                        drop=0.,
                        drop_path=0.4,
                        norm_layer=nn.LayerNorm,
                        focal_window=9,
                        focal_level=2,
                        use_layerscale=False,
                        use_checkpoint=False)

                    self.focol2 = BasicLayer(
                        dim=self.dims[2],
                        depth=2,
                        mlp_ratio=4.,
                        drop=0.,
                        drop_path=0.4,
                        norm_layer=nn.LayerNorm,
                        focal_window=9,
                        focal_level=2,
                        use_layerscale=False,
                        use_checkpoint=False)

                    self.focol3 = BasicLayer(
                        dim=self.dims[3],
                        depth=2,
                        mlp_ratio=4.,
                        drop=0.,
                        drop_path=0.4,
                        norm_layer=nn.LayerNorm,
                        focal_window=9,
                        focal_level=2,
                        use_layerscale=False,
                        use_checkpoint=False)

                    self.focol4 = BasicLayer(
                        dim=self.dims[4],
                        depth=2,
                        mlp_ratio=4.,
                        drop=0.,
                        drop_path=0.4,
                        norm_layer=nn.LayerNorm,
                        focal_window=9,
                        focal_level=2,
                        use_layerscale=False,
                        use_checkpoint=False)
                else:
                    if self.order == 2:
                        self.focol0 = BasicLayer(
                            dim=self.dims[0],
                            depth=2,
                            mlp_ratio=4.,
                            drop=0.,
                            drop_path=0.4,
                            norm_layer=nn.LayerNorm,
                            focal_window=9,
                            focal_level=2,
                            use_layerscale=False,
                            use_checkpoint=False)

                        self.focol1 = BasicLayer(
                            dim=self.dims[1],
                            depth=2,
                            mlp_ratio=4.,
                            drop=0.,
                            drop_path=0.4,
                            norm_layer=nn.LayerNorm,
                            focal_window=9,
                            focal_level=2,
                            use_layerscale=False,
                            use_checkpoint=False)


        self.AG = nn.ModuleList([
            Attention_block(F_g=self.dims[i + 1], F_l=self.dims[i + 1], F_int=self.dims[i + 1]) for i in
            range(order - 1)]
        )

        self.AG0 = Attention_block(F_g=self.dims[0], F_l=self.dims[0], F_int=self.dims[0])

        if self.order == 3:
            self.AG1 = Attention_block(F_g=self.dims[1], F_l=self.dims[1], F_int=self.dims[1])
            self.AG2 = Attention_block(F_g=self.dims[2], F_l=self.dims[2], F_int=self.dims[2])
        else:
            if self.order == 4:
                self.AG1 = Attention_block(F_g=self.dims[1], F_l=self.dims[1], F_int=self.dims[1])
                self.AG2 = Attention_block(F_g=self.dims[2], F_l=self.dims[2], F_int=self.dims[2])
                self.AG3 = Attention_block(F_g=self.dims[3], F_l=self.dims[3], F_int=self.dims[3])
            else:
                if self.order == 5:
                    self.AG1 = Attention_block(F_g=self.dims[1], F_l=self.dims[1], F_int=self.dims[1])
                    self.AG2 = Attention_block(F_g=self.dims[2], F_l=self.dims[2], F_int=self.dims[2])
                    self.AG3 = Attention_block(F_g=self.dims[3], F_l=self.dims[3], F_int=self.dims[3])
                    self.AG4 = Attention_block(F_g=self.dims[4], F_l=self.dims[4], F_int=self.dims[4])
                else:
                    if self.order == 2:
                        self.AG1 = Attention_block(F_g=self.dims[1], F_l=self.dims[1], F_int=self.dims[1])


        self.scale = s

        print('[HFConv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = self.focol0(pwa)
        x = self.AG0(g=x, x=dw_list[0])  # pwa * dw_list[0]
        x = self.focol1(self.conv0(x))
        x = self.AG1(g=x, x=dw_list[1])

        if self.order == 3:
            x = self.focol2(self.conv1(x))
            x = self.AG2(g=x, x=dw_list[2])
        else:
            if self.order == 4:
                x = self.focol2(self.conv1(x))
                x = self.AG2(g=x, x=dw_list[2])
                x = self.focol3(self.conv2(x))
                x = self.AG3(g=x, x=dw_list[3])
            else:
                if self.order == 5:
                    x = self.focol2(self.conv1(x))
                    x = self.AG2(g=x, x=dw_list[2])
                    x = self.focol3(self.conv2(x))
                    x = self.AG3(g=x, x=dw_list[3])
                    x = self.focol4(self.conv3(x))
                    x = self.AG4(g=x, x=dw_list[4])
                #else:
                #    print('Please select 2, 3, 4 and 5 order')

        x = self.proj_out(x)

        return x


class Block(nn.Module):
    """ 
    HFblock
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, HFConv=HFConv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.HFConv = HFConv(dim)  # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.HFConv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class SingleConv1(nn.Module):
    def __init__(self, in_channels, out_channels, ker_size=3, padding=1):
        super().__init__()
        self.Single_Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=ker_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.Single_Conv(x)

class Edge_aware_unit(nn.Module):
    """ 
    Lesion localization module(LL-M)
    """

    def __init__(self, in_channels, out_channels):
        super(Edge_aware_unit, self).__init__()
        self.conv1 = SingleConv1(in_channels, out_channels, ker_size=1, padding=0)
        self.conv2 = SingleConv1(out_channels, out_channels // 2, ker_size=3)
        self.conv3 = SingleConv1(out_channels//2, out_channels//2, ker_size=3)
        self.conv4 = SingleConv1(in_channels, out_channels//2, ker_size=3)

    def detect_edge_1(self, inputs, sobel_kernel):
        kernel = np.array(sobel_kernel, dtype='float32')
        kernel = kernel.reshape((1, 1, 3, 3))
        weight = Variable(torch.from_numpy(kernel)).to(device)
        edge = torch.zeros(inputs.size()[1],inputs.size()[0],inputs.size()[2],inputs.size()[3]).to(device)
        for k in range(inputs.size()[1]):
            fea_input = inputs[:,k,:,:]
            fea_input = fea_input.unsqueeze(1)
            edge_c = F.conv2d(fea_input, weight, padding=1)
            edge[k] = edge_c.squeeze(1)
        edge = edge.permute(1, 0, 2, 3)
        return edge

    def detect_edge_2(self, inputs, sobel_kernel):
        kernel = np.array(sobel_kernel, dtype='float32')
        kernel = kernel.reshape((1, 1, 5, 5))
        weight = Variable(torch.from_numpy(kernel)).to(device)
        edge = torch.zeros(inputs.size()[1],inputs.size()[0],inputs.size()[2],inputs.size()[3]).to(device)
        for k in range(inputs.size()[1]):
            fea_input = inputs[:,k,:,:]
            fea_input = fea_input.unsqueeze(1)
            edge_c = F.conv2d(fea_input, weight, padding=2)
            edge[k] = edge_c.squeeze(1)
        edge = edge.permute(1, 0, 2, 3)
        return edge

    def sobel_conv2d(self, inputs):
        edge_detect1 = self.detect_edge_1(inputs, [[2, 4, 2], [0, 0, 0], [-2, -4, -2]])
        edge_detect2 = self.detect_edge_2(inputs, [[0, 0, 0, 0, 0], [0, -2, -4, -2, 0], [-1, -4, 0, 4, 1], [0, 2, 4, 2, 0], [0, 0, 0, 0, 0]])
        edge_detect3 = self.detect_edge_1(inputs, [[4, 2, 0], [2, 0, -2], [0, -2, -4]])
        edge_detect4 = self.detect_edge_2(inputs, [[0, 0, -1, 0, 0], [0, -2, -4, 2, 0], [0, -4, 0, 4, 0], [0, -2, 4, 2, 0], [0, 0, 1, 0, 0]])
        edge_detect5 = self.detect_edge_1(inputs, [[2, 0, -2], [4, 0, -4], [2, 0, -2]])
        edge_detect6 = self.detect_edge_2(inputs, [[0, 0, 1, 0, 0], [0, -2, 4, 2, 0], [0, -4, 0, 4, 0], [0, -2, -4, 2, 0], [0, 0, -1, 0, 0]])
        edge_detect7 = self.detect_edge_1(inputs, [[0, -2, -4], [2, 0, -2], [4, 2, 0]])
        edge_detect8 = self.detect_edge_2(inputs, [[0, 0, 0, 0, 0], [0, 2, 4, 2, 0], [-1, -4, 0, 4, 1], [0, -2, -4, -2, 0], [0, 0, 0, 0, 0]])
        edge = edge_detect1+edge_detect2+edge_detect3+edge_detect4+edge_detect5+edge_detect6+edge_detect7+edge_detect8
        return edge

    def forward(self, input_f):
        conv1 = self.conv1(input_f)
        conv2 = self.conv2(conv1)
        edge_f = self.sobel_conv2d(conv2)
        conv3 = self.conv3(edge_f)
        input_f = self.conv4(input_f)
        conca = torch.cat([input_f, conv3], dim=1)
        return conca


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]


class SC_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()

        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5

        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5
        return t1, t2, t3, t4, t5

class HFUNet(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, layer_scale_init_value=1e-6, HFConv=HFConv, block=Block,
                 pretrained=None,
                 use_checkpoint=False, c_list=[32, 64, 128, 256, 512, 1024], depths=[1, 1, 1], depths_out=[1, 1, 1],
                 drop_path_rate=0.,
                 split_att='fc', mlp_ratio=4., drop_rate=0., depths_foc=[2, 2, 2], focal_levels=[2, 2, 2, 2],
                 focal_windows=[9, 9, 9, 9], norm_layer=nn.LayerNorm, use_layerscale=False, bridge=True):
        super().__init__()
        self.pretrained = pretrained
        self.use_checkpoint = use_checkpoint
        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(HFConv, list):
            HFConv = [partial(HFConv, order=2, s=1 / 3, gflayer=GlobalLocalFilter),
                      partial(HFConv, order=3, s=1 / 3, gflayer=GlobalLocalFilter),
                      partial(HFConv, order=4, s=1 / 3, h=24, w=13, gflayer=GlobalLocalFilter),
                      partial(HFConv, order=5, s=1 / 3, h=12, w=7, gflayer=GlobalLocalFilter)]
        else:
            HFConv = HFConv
            assert len(HFConv) == 3

        if isinstance(HFConv[0], str):
            HFConv = [eval(h) for h in HFConv]

        if isinstance(block, str):
            block = eval(block)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_foc))]

        self.encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1)

        self.encoder4 = nn.Sequential(
            *[block(dim=c_list[2], drop_path=dp_rates[0 + j],
                    layer_scale_init_value=layer_scale_init_value, HFConv=HFConv[2]) for j in range(depths[0])],
            Edge_aware_unit(c_list[2], c_list[2]),
            nn.Conv2d(c_list[2], c_list[3], 3, stride=1, padding=1),
        )

        self.encoder5 = nn.Sequential(
            *[block(dim=c_list[3], drop_path=dp_rates[1 + j],
                    layer_scale_init_value=layer_scale_init_value, HFConv=HFConv[2]) for j in range(depths[1])],
            Edge_aware_unit(c_list[3], c_list[3]),
            nn.Conv2d(c_list[3], c_list[4], 3, stride=1, padding=1),
        )

        self.encoder6 = nn.Sequential(
            *[block(dim=c_list[4], drop_path=dp_rates[2 + j],
                    layer_scale_init_value=layer_scale_init_value, HFConv=HFConv[2]) for j in range(depths[2])],
            Edge_aware_unit(c_list[4], c_list[4]),
            nn.Conv2d(c_list[4], c_list[5], 3, stride=1, padding=1),
        )

        # build Bottleneck layers
        self.ConvMixer = ConvMixerBlock(dim=c_list[5], depth=7, k=7)
        # Skip-connection
        self.mdag4 = MDAG(channel=c_list[4])
        self.mdag3 = MDAG(channel=c_list[3])
        self.mdag2 = MDAG(channel=c_list[2])
        self.mdag1 = MDAG(channel=c_list[1])
        self.mdag0 = MDAG(channel=c_list[0])

        if bridge:
            self.scab = SC_Att_Bridge()
            print('SC_Att_Bridge was used')

        dp_rates_out = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_out))]

        self.decoder1 = nn.Sequential(
            nn.Conv2d(c_list[5], c_list[4], 3, stride=1, padding=1),
            Edge_aware_unit(c_list[4], c_list[4]),
            *[block(dim=c_list[4], drop_path=dp_rates_out[2 + j],
                    layer_scale_init_value=layer_scale_init_value, HFConv=HFConv[2]) for j in range(depths_out[2])],
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[3], 3, stride=1, padding=1),
            Edge_aware_unit(c_list[3], c_list[3]),
            *[block(dim=c_list[3], drop_path=dp_rates_out[1 + j],
                    layer_scale_init_value=layer_scale_init_value, HFConv=HFConv[2]) for j in range(depths_out[1])],
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[2], 3, stride=1, padding=1),
            Edge_aware_unit(c_list[2], c_list[2]),
            *[block(dim=c_list[2], drop_path=dp_rates_out[0 + j],
                    layer_scale_init_value=layer_scale_init_value, HFConv=HFConv[2]) for j in range(depths_out[0])],
        )

        self.decoder4 = nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1)

        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.ebn6 = nn.GroupNorm(4, c_list[5])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        t5 = self.mdag4(x=t5)
        t4 = self.mdag3(x=t4)
        t3 = self.mdag2(x=t3)
        t2 = self.mdag1(x=t2)
        t1 = self.mdag0(x=t1)

        out = F.gelu((self.ebn6(self.encoder6(out))))
        out = self.ConvMixer(out)


        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        out5 = torch.add(out5, t5)

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))
        out4 = torch.add(out4, t4)

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))
        out3 = torch.add(out3, t3)

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))
        out2 = torch.add(out2, t2)

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))
        out1 = torch.add(out1, t1)

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)

        return torch.sigmoid(out0)

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GlobalLocalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, padding=0, bias=False, groups=dim // 2),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2),
        )
        self.conv2d = nn.Conv2d(dim, dim // 2, kernel_size=1, padding=0, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1 = self.dw(x)

        x2 = x.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',
                                   align_corners=True).permute(1, 2, 3, 0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')
        x2 = self.conv2d(x2)

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, C, a, b)
        x = self.post_norm(x)
        return x


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FocalModulation(nn.Module):
    """ Focal Modulation

    Args:
        dim (int): Number of input channels.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int, default=2): Step to increase the focal window
        use_postln (bool, default=False): Whether use post-modulation layernorm
    """

    def __init__(self, dim, proj_drop=0., focal_level=2, focal_window=7, focal_factor=2, use_postln=False):

        super().__init__()
        self.dim = dim

        # specific args for focalv3
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln = use_postln

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=True)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        if self.use_postln:
            self.ln = nn.LayerNorm(dim)

        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim,
                              padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )

    def forward(self, x):
        """ Forward function.

        Args:
            x: input features with shape of (B, H, W, C)
        """
        B, nH, nW, C = x.shape
        x = self.f(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        x_out = q * self.h(ctx_all)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln:
            x_out = self.ln(x_out)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


class FocalModulationBlock(nn.Module):
    """ Focal Modulation Block.

    Args:
        dim (int): Number of input channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int): number of focal levels
        focal_window (int): focal kernel size at level 1
    """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 focal_level=2, focal_window=9, use_layerscale=False, layerscale_value=1e-4):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_layerscale = use_layerscale

        self.norm1 = norm_layer(dim)
        self.modulation = FocalModulation(
            dim, focal_window=self.focal_window, focal_level=self.focal_level, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if self.use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, C, H, W = x.shape
        x = x.view(B, H, W, C)
        x = self.norm1(x)
        shortcut = x.view(B, H * W, C)
        x = x.view(B, H, W, C)

        # FM
        x = self.modulation(x).view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = x.view(B, C, H, W)

        return x


class BasicLayer(nn.Module):
    """ A basic focal modulation layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        use_conv_embed (bool): Use overlapped convolution for patch embedding or now. Default: False
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self,
                 dim,
                 depth,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 focal_window=9,
                 focal_level=2,
                 use_conv_embed=False,
                 use_layerscale=False,
                 use_checkpoint=False
                 ):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            FocalModulationBlock(
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                focal_window=focal_window,
                focal_level=focal_level,
                use_layerscale=use_layerscale,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                patch_size=2,
                in_chans=dim, embed_dim=2 * dim,
                use_conv_embed=use_conv_embed,
                norm_layer=norm_layer,
                is_stem=False
            )

        else:
            self.downsample = None

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


# bottleneck
class ConvMixerBlock(nn.Module):
    def __init__(self, dim=256, depth=7, k=7):
        super(ConvMixerBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(dim, dim, kernel_size=(k, k), groups=dim, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )

    def forward(self, x):
        x = self.block(x)
        return x

class MDAG(nn.Module):
    """
    Multi-dilation attention gate
    """

    def __init__(self, channel, k_size=3, dilated_ratio=[7, 5, 2, 1]):
        super(MDAG, self).__init__()
        self.channel = channel
        self.mda0 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=k_size, stride=1,
                      padding=(k_size + (k_size - 1) * (dilated_ratio[0] - 1)) // 2,
                      dilation=dilated_ratio[0]),
            nn.BatchNorm2d(self.channel))
        self.mda1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=k_size, stride=1,
                      padding=(k_size + (k_size - 1) * (dilated_ratio[1] - 1)) // 2,
                      dilation=dilated_ratio[1]),
            nn.BatchNorm2d(self.channel))
        self.mda2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=k_size, stride=1,
                      padding=(k_size + (k_size - 1) * (dilated_ratio[2] - 1)) // 2,
                      dilation=dilated_ratio[2]),
            nn.BatchNorm2d(self.channel))
        self.mda3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=k_size, stride=1,
                      padding=(k_size + (k_size - 1) * (dilated_ratio[3] - 1)) // 2,
                      dilation=dilated_ratio[3]),
            nn.BatchNorm2d(self.channel))
        self.voteConv = nn.Sequential(
            nn.Conv2d(self.channel * 4, self.channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.channel),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.AG = Attention_block(F_g=self.channel, F_l=self.channel, F_int=self.channel)

    def forward(self, x):
        x1 = self.mda0(x)
        x2 = self.mda1(x)
        x3 = self.mda2(x)
        x4 = self.mda3(x)
        _x = self.relu(torch.cat((x1, x2, x3, x4), dim=1))
        _x = self.voteConv(_x)
        x = x + x * _x
        return x

