import torch
import torch.nn as nn
import torch.nn.functional as nnf
from typing import Union
import numpy as np

 
class conv(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        has_bn=True, 
        has_relu=True,
        **kwargs
        ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class deconv(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        has_bn=True, 
        has_relu=False,
        **kwargs
        ):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
            )
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.deconv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class resnext_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups, has_proj=False):
        super().__init__()
        bottleneck = out_channels//4
        assert (bottleneck % groups == 0) and (bottleneck / groups) % 4 == 0, (bottleneck, groups)
        self.conv_1x1_shrink = conv(in_channels, bottleneck, kernel_size=1, padding=0)
        self.conv_3x3        = conv(bottleneck,  bottleneck, kernel_size=3, stride=stride, groups=groups)
        self.conv_1x1_expand = conv(bottleneck,  out_channels, kernel_size=1, padding=0, has_relu=False) 

        self.has_proj = has_proj
        if self.has_proj:
            if stride == 2:
                self.dsp = nn.AvgPool2d(kernel_size=2, stride=2)
            self.shortcut = conv(in_channels, out_channels, kernel_size=1, padding=0, has_relu=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        proj = x
        if self.has_proj:
            if hasattr(self, "dsp"):
                proj = self.dsp(proj)
            proj = self.shortcut(proj)
        x = self.conv_1x1_shrink(x)
        x = self.conv_3x3(x)
        x = self.conv_1x1_expand(x)
        x = x + proj
        x = self.relu(x)
        return x


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, ks):
        super().__init__()
        self.conv00 = conv(in_channels,  out_channels, kernel_size=(1,ks), padding=(0, ks//2))
        self.conv01 = conv(out_channels, out_channels, kernel_size=(ks,1), padding=(ks//2, 0), has_relu=False)
        self.conv10 = conv(in_channels,  out_channels, kernel_size=(ks,1), padding=(ks//2, 0))
        self.conv11 = conv(out_channels, out_channels, kernel_size=(1,ks), padding=(0, ks//2), has_relu=False)
    
    def forward(self, x):
        x0 = self.conv00(x)
        x0 = self.conv01(x0)
        x1 = self.conv10(x)
        x1 = self.conv11(x1)
        x  = x0 + x1
        return x


class refine_block(nn.Module):
    def __init__(self, in_channels, out_channels, ks):
        super().__init__()
        self.refine0 = conv(in_channels,  out_channels, kernel_size=ks, padding=ks//2)
        self.refine1 = conv(out_channels, out_channels, kernel_size=ks, padding=ks//2, has_relu=False)
    def forward(self, data):
        x = self.refine0(data)
        x = self.refine1(x)
        x = x + data
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, base_ch=32):
        super().__init__()
        # 1
        self.conv00 = conv(in_channels, base_ch//2, stride=2)
        
        # 1/2
        self.g03_0 = resnext_block(base_ch//2, base_ch*2, stride=2, groups=2,  has_proj=True)
        self.g03_1 = resnext_block(base_ch*2,  base_ch*2, stride=1,  groups=2, has_proj=False)
        self.g03_2 = resnext_block(base_ch*2,  base_ch*2, stride=1,  groups=2, has_proj=False)
        self.p03   = nn.AvgPool2d(kernel_size=32, stride=32)
        
        # 1/4
        self.g3_0 = resnext_block(base_ch*2,  base_ch*4, stride=2,  groups=2, has_proj=True)
        self.g3_1 = resnext_block(base_ch*4,  base_ch*4, stride=1,  groups=2, has_proj=False)
        self.g3_2 = resnext_block(base_ch*4,  base_ch*4, stride=1,  groups=2, has_proj=False)
        self.g3_3 = resnext_block(base_ch*4,  base_ch*4, stride=1,  groups=2, has_proj=False)
        self.p3   = nn.AvgPool2d(kernel_size=16, stride=16)

        # 1/8
        self.g4_0 = resnext_block(base_ch*4,  base_ch*8, stride=2,  groups=4, has_proj=True)
        self.g4_1 = resnext_block(base_ch*8,  base_ch*8, stride=1,  groups=4, has_proj=False)
        self.g4_2 = resnext_block(base_ch*8,  base_ch*8, stride=1,  groups=4, has_proj=False)
        self.g4_3 = resnext_block(base_ch*8,  base_ch*8, stride=1,  groups=4, has_proj=False)
        self.g4_4 = resnext_block(base_ch*8,  base_ch*8, stride=1,  groups=4, has_proj=False)
        self.g4_5 = resnext_block(base_ch*8,  base_ch*8, stride=1,  groups=4, has_proj=False)
        self.g4_6 = resnext_block(base_ch*8,  base_ch*8, stride=1,  groups=4, has_proj=False)
        self.p4   = nn.AvgPool2d(kernel_size=8, stride=8)

        # 1/16
        self.g5_0 = resnext_block(base_ch*8,  base_ch*16, stride=2,  groups=8, has_proj=True)
        self.g5_1 = resnext_block(base_ch*16,  base_ch*16, stride=1,  groups=8, has_proj=False)
        self.g5_2 = resnext_block(base_ch*16,  base_ch*16, stride=1,  groups=8, has_proj=False)
        self.g5_3 = resnext_block(base_ch*16,  base_ch*16, stride=1,  groups=8, has_proj=False)
        self.g5_4 = resnext_block(base_ch*16,  base_ch*16, stride=1,  groups=8, has_proj=False)
        self.g5_5 = resnext_block(base_ch*16,  base_ch*16, stride=1,  groups=8, has_proj=False)
        self.g5_6 = resnext_block(base_ch*16,  base_ch*16, stride=1,  groups=8, has_proj=False)
        self.p5   = nn.AvgPool2d(kernel_size=4, stride=4)

        # 1/32
        self.g6_0 = resnext_block(base_ch*16,  base_ch*32,  stride=2,  groups=8, has_proj=True)
        self.g6_1 = resnext_block(base_ch*32,  base_ch*32,  stride=1,  groups=8, has_proj=False)
        self.g6_2 = resnext_block(base_ch*32,  base_ch*32,  stride=1,  groups=8, has_proj=False)
        self.g6_3 = resnext_block(base_ch*32,  base_ch*32,  stride=1,  groups=8, has_proj=False)
        self.g6_4 = resnext_block(base_ch*32,  base_ch*32,  stride=1,  groups=8, has_proj=False)
        self.g6_5 = resnext_block(base_ch*32,  base_ch*32,  stride=1,  groups=8, has_proj=False)
        self.g6_6 = resnext_block(base_ch*32,  base_ch*32,  stride=1,  groups=8, has_proj=False)
        self.p6   = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 1
        x0 = self.conv00(x)
        # 1/2
        x01 = self.g03_0(x0)
        x01 = self.g03_1(x01)
        x01 = self.g03_2(x01)
        pool01 = self.p03(x0)
        # 1/4
        x1 = self.g3_0(x01)
        x1 = self.g3_1(x1)
        x1 = self.g3_2(x1)
        x1 = self.g3_3(x1)
        pool1 = self.p3(x01)
        # 1/8
        x2 = self.g4_0(x1)
        x2 = self.g4_1(x2)
        x2 = self.g4_2(x2)
        x2 = self.g4_3(x2)
        x2 = self.g4_4(x2)
        x2 = self.g4_5(x2)
        x2 = self.g4_6(x2)
        pool2 = self.p4(x1)        
        # 1/16
        x3 = self.g5_0(x2)
        x3 = self.g5_1(x3)
        x3 = self.g5_2(x3)
        x3 = self.g5_3(x3)
        x3 = self.g5_4(x3)
        x3 = self.g5_5(x3)
        x3 = self.g5_6(x3)
        pool3 = self.p5(x2)    
        # 1/32
        x4 = self.g6_0(x3)
        x4 = self.g6_1(x4)
        x4 = self.g6_2(x4)
        x4 = self.g6_3(x4)
        x4 = self.g6_4(x4)
        x4 = self.g6_5(x4)
        x4 = self.g6_6(x4)
        pool4 = self.p6(x3)    

        x4 = torch.cat([pool01, pool1, pool2, pool3, pool4, x4], dim=1)
        return x4, x3, x2, x1, x01, x0


class UNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = Encoder(in_channels)

        is_conv_block = [True, True, True, False, False, False]
        # base_ch*2 + base_ch*4 + base_ch*8 + base_ch*16 + base_ch*32
        encdr_ch = [2000, 512, 256, 128, 64, 16]
        in_ch  = [8*16, 8*16, 8*8, 8*2, 8, 8]
        out_ch = [8*16, 8*8, 8*2, 8, 8, 8]
        for idx, (encdr_channels, in_channels, out_channels, _is_conv_block) in enumerate(zip(encdr_ch, in_ch, out_ch, is_conv_block)):
            if _is_conv_block:
                score  = conv_block(encdr_channels, in_channels, 3)
            else:
                score  = conv(encdr_channels, in_channels, has_relu=False)
            setattr(self, f"score{idx}", score)

            refine = refine_block(in_channels, in_channels, 3)
            setattr(self, f"refine{idx}", refine)

            if idx < len(is_conv_block) - 1:
                resize = deconv(in_channels, out_channels, 2, stride=2, padding=0, has_relu=False)
            else:
                resize = deconv(in_channels, 1, 2, stride=2, padding=0, has_bn=False, has_relu=False)
            setattr(self, f"resize{idx}", resize)
            
        # valid depth-estsimation branch
        is_conv_block = [True, True, True, False, False, False]
        encdr_ch = [2000, 512, 256, 128, 64, 16]
        in_ch  = [8*16, 8*16, 8*8, 8*2, 8, 8]
        out_ch = [8*16, 8*8, 8*2, 8, 8, 8]
        for idx, (encdr_channels, in_channels, out_channels, _is_conv_block) in enumerate(zip(encdr_ch, in_ch, out_ch, is_conv_block)):
            if _is_conv_block:
                valid_score  = conv_block(encdr_channels, in_channels, 3)
            else:
                valid_score  = conv(encdr_channels, in_channels, has_relu=False)
            setattr(self, f"valid_score{idx}", valid_score)

            valid_refine = refine_block(in_channels, in_channels, 3)
            setattr(self, f"valid_refine{idx}", valid_refine)

            if idx < len(is_conv_block) - 1:
                valid_resize = deconv(in_channels, out_channels, 2, stride=2, padding=0, has_relu=False)
            else:
                valid_resize = deconv(in_channels, 1, 2, stride=2, padding=0, has_bn=False, has_relu=False)
            setattr(self, f"valid_resize{idx}", valid_resize)


    def forward(self, data):
        conv4, conv3, conv2, conv1, conv01, conv0 = self.encoder(data)
        blocks = [conv4, conv3, conv2, conv1, conv01, conv0]

        last_fm = None
        up_former = None
        for idx in range(len(blocks)):
            up_latter = getattr(self, f"score{idx}")(blocks[idx])
            if idx > 0:
                up_latter = up_latter + up_former
            up_former = getattr(self, f"refine{idx}")(up_latter)
            up_former = getattr(self, f"resize{idx}")(up_former)
            if idx == len(blocks) -2:
                last_fm = up_former

        pred = up_former
        pred = nnf.relu(pred)

        up_former = None
        for idx in range(len(blocks)):
            up_latter = getattr(self, f"valid_score{idx}")(blocks[idx])
            if idx > 0:
                up_latter = up_latter + up_former
            up_former = getattr(self, f"valid_refine{idx}")(up_latter)
            up_former = getattr(self, f"valid_resize{idx}")(up_former)

        valid = up_former
        valid = nnf.sigmoid(valid)

        return pred, valid, last_fm


class FullyConnected(torch.nn.Module):
    '''
    Fully connected layer

    Arg(s):
        in_channels : int
            number of input neurons
        out_channels : int
            number of output neurons
        dropout_rate : float
            probability to use dropout
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 dropout_rate=0.00):
        super(FullyConnected, self).__init__()

        self.fully_connected = torch.nn.Linear(in_features, out_features)

        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.fully_connected.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.fully_connected.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.fully_connected.weight)

        self.activation_func = activation_func

        if dropout_rate > 0.00 and dropout_rate <= 1.00:
            self.dropout = torch.nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        fully_connected = self.fully_connected(x)

        if self.activation_func is not None:
            fully_connected = self.activation_func(fully_connected)

        if self.dropout is not None:
            return self.dropout(fully_connected)
        else:
            return fully_connected


def _activation_func(activation_fn):
    '''
    Select activation function
    Arg(s):
        activation_fn : str
            name of activation function
    '''

    if 'linear' in activation_fn:
        return None
    elif 'leaky_relu' in activation_fn:
        return torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)
    elif 'relu' in activation_fn:
        return torch.nn.ReLU()
    elif 'elu' in activation_fn:
        return torch.nn.ELU()
    elif 'sigmoid' in activation_fn:
        return torch.nn.Sigmoid()
    else:
        raise ValueError('Unsupported activation function: {}'.format(activation_fn))


class FullyConnectedEncoder(torch.nn.Module):
    '''
    Fully connected encoder
    Arg(s):
        input_channels : int
            number of input channels
        n_neurons : list[int]
            number of filters to use per layer
        latent_size : int
            number of output neuron
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function after convolution
    '''

    def __init__(self,
                 input_channels=3,
                 n_neurons=[32, 64, 96, 64, 32],
                 latent_size=8,
                 weight_initializer='kaiming_uniform',
                 ):
        super(FullyConnectedEncoder, self).__init__()

        activation_func = torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)

        self.mlp = torch.nn.Sequential(
            FullyConnected(
                in_features=input_channels,
                out_features=n_neurons[0],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            FullyConnected(
                in_features=n_neurons[0],
                out_features=n_neurons[1],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            FullyConnected(
                in_features=n_neurons[1],
                out_features=n_neurons[2],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            FullyConnected(
                in_features=n_neurons[2],
                out_features=n_neurons[3],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            FullyConnected(
                in_features=n_neurons[3],
                out_features=n_neurons[4],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            FullyConnected(
                in_features=n_neurons[4],
                out_features=latent_size,
                weight_initializer=weight_initializer,
                activation_func=activation_func,))

    def forward(self, x):

        return self.mlp(x)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.G1  = UNet(4)        # 1/4
        self.G2  = UNet(12)       # 1/2      
        self.G3  = UNet(12)       # 1      
        
        self.calibrated_params_encoder = FullyConnectedEncoder(input_channels=3,)

    def forward(self, img, radar, radar_pts, valid_radar_pts_cnts):

        B,K,C = radar_pts.shape
        radar_pts = radar_pts.reshape(B*K, -1)
        radar_pts_emb = self.calibrated_params_encoder(radar_pts)
        radar_pts_emb = radar_pts_emb.reshape(B,K,radar_pts_emb.shape[-1])
        
        valid_radar_pts_emb = []
        for _radar_pts_emb, _valid_radar_pts_cnts in zip(radar_pts_emb, valid_radar_pts_cnts):
            valid_radar_pts_emb.append(_radar_pts_emb[:_valid_radar_pts_cnts,:].mean(axis=0))
        valid_radar_pts_emb = torch.stack(valid_radar_pts_emb, dim=0)[...,None,None]

        img_dw2 = nnf.interpolate(img, scale_factor=0.5,  mode="bilinear")        
        radar_dw2 = self.downsample_depthmap(radar, 0.5)
        data_dw2 =  torch.concat((img_dw2,radar_dw2), dim=1)
        
        img_dw4 = nnf.interpolate(img, scale_factor=0.25, mode="bilinear")
        radar_dw4 = self.downsample_depthmap(radar, 0.25)
        data_dw4 =  torch.concat((img_dw4,radar_dw4),dim=1)
        
        pred1_dw4, valid1_dw4, fm1_dw8 = self.G1(data_dw4)
        
        fm1_dw8 = fm1_dw8 + valid_radar_pts_emb
        fm1_dw2  = nnf.interpolate(fm1_dw8, scale_factor=4., mode="bilinear")

        data_dw2 = torch.concat([data_dw2, fm1_dw2], dim=1)
        pred2_dw2, valid2_dw2, fm2_dw4 = self.G2(data_dw2)
        
        fm2_dw4 = fm2_dw4 + valid_radar_pts_emb
        fm2 = nnf.interpolate(fm2_dw4, scale_factor=4., mode="bilinear")

        data = torch.concat((img, radar, fm2), axis=1)
        pred3, valid3, _ = self.G3(data)
        
        return [pred1_dw4, pred2_dw2, pred3], [valid1_dw4, valid2_dw2, valid3]

    @staticmethod
    def make_torch_tensor(data:Union[np.ndarray, torch.Tensor], device, dtype) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            if data.dtype == np.uint16:
                data = data.astype(np.float32)
            data = torch.from_numpy(data)
        if data.device != device:
            data = data.to(device)
        if data.dtype != dtype:
            data = data.to(dtype)
        return data

    @staticmethod
    def padding(data, int=256, dim=0):
        # dim: 0 for 2 axis, 1 for height, 2 for width
        if dim == 0:
            B, C, H, W = data.shape
            new_h, new_w = (H//int + 1) * int, (W//int + 1) * int
            new_data = torch.zeros((B, C, new_h, new_w), device=data.device)
            new_data[:, :, :H, :W] = data
        elif dim == 1:
            B, C, H, W = data.shape
            new_h, new_w = (H//int + 1) * int, (W//int + 1) * int
            new_data = torch.zeros((B, C, new_h, W), device=data.device)
            new_data[:, :, :H, :W] = data
        else:
            B, C, H, W = data.shape
            new_h, new_w = (H//int + 1) * int, (W//int + 1) * int
            new_data = torch.zeros((B, C, H, new_w), device=data.device)
            new_data[:, :, :H, :W] = data
        return new_data

    @property
    def device(self, ):
        device_set = set([p.device for p in self.parameters()])
        assert len(device_set) == 1
        return device_set.pop()

    @property
    def dtype(self, ):
        dtype_set = set([p.dtype for p in self.parameters()])
        assert len(dtype_set) == 1
        return dtype_set.pop() 


    def downsample_depthmap(self, inp, factor):
        assert factor<=1
        n,c,h,w = inp.shape
        new_h, new_w = int(h*factor), int(w*factor)
        out = torch.zeros((n,c,new_h,new_w), device=self.device)

        nonzero_idx = list(torch.nonzero(inp, as_tuple=True))
        nonzero_arr = inp[nonzero_idx]
        nonzero_idx[2] = (nonzero_idx[2]*factor).long()
        nonzero_idx[3] = (nonzero_idx[3]*factor).long()
        out[nonzero_idx] = nonzero_arr
        return out


    def forward_train(self, mini_batch_data:dict):
        img              = self.padding(self.make_torch_tensor(mini_batch_data["img"], self.device, self.dtype))
        label            = self.padding(self.make_torch_tensor(mini_batch_data["label"], self.device, self.dtype))
        label_mask       = self.padding(self.make_torch_tensor(mini_batch_data['label_mask'], self.device, self.dtype))
        radar            = self.padding(self.make_torch_tensor(mini_batch_data['radar'], self.device, self.dtype))
        valid_label      = self.padding(self.make_torch_tensor(mini_batch_data['seg_mask_roi'], self.device, self.dtype))
        radar_pts      = self.make_torch_tensor(mini_batch_data['radar_pts'], self.device, self.dtype)
        valid_radar_pts_cnts      = self.make_torch_tensor(mini_batch_data['valid_radar_pts_cnt'], self.device, torch.long)

        pred_pyramid, valid_pyramid = self.forward(img, radar, radar_pts, valid_radar_pts_cnts)

        losses, monitors = [], {}
        for i in range(len(pred_pyramid)):

            scale = 2 ** (len(pred_pyramid) - 1 - i)

            # pred
            pred_tmp  = pred_pyramid[i]
            valid_tmp = valid_pyramid[i]

            # label
            label_tmp       = self.downsample_depthmap(label, 1./scale)
            valid_label_tmp = nnf.interpolate(valid_label, size=(label.shape[2]//scale,label.shape[3]//scale), mode="nearest")

            # valid mask
            mask_tmp       = self.downsample_depthmap(label_mask, 1./scale)
            
            # cal loss
            valid_loss = torch.mean(torch.abs(valid_tmp - valid_label_tmp))
            monitors[f'train_valid_loss_{i}'] = valid_loss
            losses.append(valid_loss / scale)

            train_loss = get_loss_l1_smooth(pred_tmp, label_tmp, mask_tmp)
            monitors['train_loss_%d' % i] = train_loss
            losses.append(train_loss / scale)  # Add weight here

        loss = sum(losses)
        return loss, monitors


    def forward_eval(        
        self, 
        mini_batch_data: dict
        ):
        
        B,C, H, W = mini_batch_data["img"].shape
        img              = self.padding(self.make_torch_tensor(mini_batch_data["img"],        self.device, self.dtype))
        label            = self.padding(self.make_torch_tensor(mini_batch_data["label"],      self.device, self.dtype))
        label_mask       = self.padding(self.make_torch_tensor(mini_batch_data['label_mask'], self.device, self.dtype))
        radar            = self.padding(self.make_torch_tensor(mini_batch_data['radar'],      self.device, self.dtype))
        valid_label      = self.padding(self.make_torch_tensor(mini_batch_data['seg_mask_roi'],      self.device, self.dtype))
        radar_pts      = self.make_torch_tensor(mini_batch_data['radar_pts'], self.device, self.dtype)
        valid_radar_pts_cnts      = self.make_torch_tensor(mini_batch_data['valid_radar_pts_cnt'], self.device, torch.long)

        idx = torch.randint(low=0, high=img.shape[0], size=(1, ))[0]
        img              = img[idx, None, ...]
        label            = label[idx, None, ...]
        label_mask       = label_mask[idx, None, ...]
        radar            = radar[idx, None, ...]
        valid_label      = valid_label[idx, None, ...]
        radar_pts = radar_pts[idx, None, ...]
        valid_radar_pts_cnts = valid_radar_pts_cnts[idx, None, ...]

        with torch.no_grad():
            pred_pyramid, valid_pyramid = self.forward(img, radar, radar_pts, valid_radar_pts_cnts)
        
        return {
            "pred":             pred_pyramid[-1][:,:, :H,:W],
            'pred_mask':        valid_pyramid[1][:,:, :H//2,:W//2],
            "radar":            radar[:,:, :H,:W],
            "img":              img[:,:, :H,:W],
            "label":            label[:,:, :H,:W],
            "label_mask":       label_mask[:,:, :H,:W],
            "valid_label":      valid_label[:,:, :H,:W],
        }

    def forward_test(        
        self, 
        img,
        radar,
        radar_pts,
        valid_radar_pts_cnts
        ):
        B, C, H, W = img.shape
        img                  = self.padding(self.make_torch_tensor(img, self.device, self.dtype))
        radar                = self.padding(self.make_torch_tensor(radar, self.device, self.dtype))
        radar_pts            = self.make_torch_tensor(radar_pts, self.device, self.dtype)
        valid_radar_pts_cnts = self.make_torch_tensor(valid_radar_pts_cnts, self.device, torch.long)

        pred_pyramid, valid_pyramid = self.forward(img, radar, radar_pts, valid_radar_pts_cnts)
        
        return pred_pyramid[-1][:,:, :H,:W].detach().cpu().numpy(), valid_pyramid[-1][:,:, :H,:W].detach().cpu().numpy()
    

def get_loss_l1_smooth(pred, label, mask):
    pred        = pred.reshape(pred.shape[0], -1)
    label       = label.reshape(label.shape[0], -1)
    mask        = mask.reshape(mask.shape[0], -1)
    diff        = (pred - label) * mask
    smooth_mask = (diff.abs() < 1.0).float()
    value       = (0.5 * diff ** 2) * smooth_mask + (diff.abs() - 0.5) * (1.0 - smooth_mask)
    serr        = value.sum(axis=1).mean() / mask.sum()
    L           = 1.0  #  label.partial_shape[1]
    loss        = serr / L
    return loss


if __name__ == "__main__":
    from thop import profile

    net = Network()
    total = sum(p.numel() for p in net.parameters())
    print("Total params: %.2fM" % (total/1e6))

    img = torch.randn(1, 3, 1024, 1792)
    radar = torch.randn(1, 1, 1024, 1792)
    radar_pts = torch.randn(1, 50, 3)
    valid_radar_pts_num = torch.randint(20, 50, size=(1,1))
    flops, params = profile(net, inputs=(img, radar, radar_pts, valid_radar_pts_num))
    print(f"Flops: {flops / 1e9} G")