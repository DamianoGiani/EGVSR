import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F
from SRNET import BicubicUpsample
import warnings
from functools import partial
from collections import OrderedDict
import os.path as osp
import cv2
import numpy as np
import sys

torch.cuda.empty_cache()

def float32_to_uint8(inputs):
    """ Convert np.float32 array to np.uint8

        Parameters:
            :param input: np.float32, (NT)CHW, [0, 1]
            :return: np.uint8, (NT)CHW, [0, 255]
    """
    return np.uint8(np.clip(np.round(inputs * 255), 0, 255))


def toimage(x,i,pathRes):
    with torch.no_grad():        
        np_arr=x.squeeze(0).cpu().numpy().transpose(1, 2, 0)       
        img3=float32_to_uint8(np_arr)
        img = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        cv2.imwrite(pathRes+str(i)+'.png', img)
        print('frame '+str(i)+' recostructed')

def load_checkpoint(fpath):
    if fpath is None:
        raise ValueError('File path is None')
    fpath = osp.abspath(osp.expanduser(fpath))
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


class ResidualBlock(nn.Module):
    """ Residual block without batch normalization
    """

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

    def forward(self, x):
        out = self.conv(x) + x

        return out


class SRNet(nn.Module):
    """ Reconstruction & Upsampling network
    """

    def __init__(self, in_nc=6, out_nc=3, nf=64, nb=16, upsample_func=None,
                 scale=4):
        super(SRNet, self).__init__()

        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d((scale ** 2 + 1) * in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # residual blocks
        self.resblocks = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])

        # upsampling
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(inplace=True))

        self.conv_up_cheap = nn.Sequential(
            nn.PixelShuffle(4),
            nn.ReLU(inplace=True))

        # output conv.
        self.conv_out = nn.Conv2d(4, out_nc, 3, 1, 1, bias=True)

        # upsampling function
        self.upsample_func = upsample_func

    def forward(self, x):
        """ lr_curr: the current lr data in shape nchw
            hr_prev_tran: the previous transformed hr_data in shape n(4*4*c)hw
        """

        out = self.conv_in(x)
        
        out = self.resblocks(out)
       
        out = self.conv_up_cheap(out)
        out = self.conv_out(out)
        # out += self.upsample_func(lr_curr)

        return out


class BaseSequenceGenerator(nn.Module):
    def __init__(self):
        super(BaseSequenceGenerator, self).__init__()

    def generate_dummy_input(self, lr_size):
        """ use for compute per-step FLOPs and speed
            return random tensors that can be taken as input of <forward>
        """
        return None

    def forward(self, *args, **kwargs):
        """ forward pass for a singe frame
        """
        pass

    def forward_sequence(self, lr_data):
        """ forward pass for a whole sequence (for training)
        """
        pass

    def infer_sequence(self, lr_data, device):
        """ infer for a whole sequence (for inference)
        """
        pass


class FNet(nn.Module):
    """ Optical flow estimation network
    """

    def __init__(self, in_nc):
        super(FNet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(2 * in_nc, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.flow = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1, bias=True))

    def forward(self, x1, x2):
        """ Compute optical flow from x1 to x2
        """

        out = self.encoder1(torch.cat([x1, x2], dim=1))
        out = self.encoder2(out)
        out = self.encoder3(out)
        out = F.interpolate(
            self.decoder1(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder2(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder3(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.tanh(self.flow(out)) * 24  # 24 is the max velocity

        return out


class FRNet(BaseSequenceGenerator):
    """ Frame-recurrent network proposed in https://arxiv.org/abs/1801.04590
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, scale=4):
        super(FRNet, self).__init__()

        self.scale = scale

        # get upsampling function according to the degradation mode
        self.upsample_func = BicubicUpsample(scale_factor=4)

        # define fnet & srnet
        self.fnet = FNet(in_nc)
        self.srnet = SRNet(in_nc, out_nc, nf, nb, self.upsample_func)

    def forward(self, x):
        """ lr_curr: the current lr data in shape nchw
            hr_prev_tran: the previous transformed hr_data in shape n(4*4*c)hw
        """

        out = self.srnet.conv_in(x)
        
        out = self.srnet.resblocks(out)
       
        out = self.srnet.conv_up_cheap(out)
        out = self.srnet.conv_out(out)
        # out += self.upsample_func(lr_curr)

        return out


def load_pretrained_weights(model, weight_path):
    checkpoint = load_checkpoint(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
                format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                    format(discarded_layers)
            )


# frnet=FRNet(3,3,64,16,4)
# print(frnet)


def space_to_depth(x, scale=4):
    """ Equivalent to tf.space_to_depth()
    """

    n, c, in_h, in_w = x.size()
    out_h, out_w = in_h // scale, in_w // scale

    x_reshaped = x.reshape(n, c, out_h, scale, out_w, scale)
    x_reshaped = x_reshaped.permute(0, 3, 5, 1, 2, 4)
    output = x_reshaped.reshape(n, scale * scale * c, out_h, out_w)

    return output


with torch.no_grad():
    a=(sys.argv)    
    frnet = FRNet(3, 3, 64, 16, 4)
    upsample_func = BicubicUpsample(scale_factor=4)
    if a[1]=='001':
      load_pretrained_weights(frnet, '/content/EGVSR/pretrained_models/MyG_iter12000.pth')
      pathRes='/content/results/MyFirstMod/MyG_iter12000/frame'
    else: 
      load_pretrained_weights(frnet, '/content/EGVSR/pretrained_models/EGVSR_iter420000.pth')
      pathRes='/content/results/MyFirstMod/EGVSR_iter420000/frame'
    open_file = open('/content/EGVSR/backgroundLR.pkl', "rb")
    lr1 = pickle.load(open_file)
    open_file1 = open('/content/EGVSR/foregroundLR.pkl', "rb")
    lr2 = pickle.load(open_file1)
    open_file2 = open('/content/EGVSR/background.pkl', "rb")
    hr_warp1 = pickle.load(open_file2)
    open_file3 = open('/content/EGVSR/foreground.pkl', "rb")
    hr_warp2 = pickle.load(open_file3)
    
    if torch.cuda.is_available():
        frnet.cuda()
    for i in range(len(lr1)):
       
        v = torch.add(lr1[i], lr2[i])
        
        k = torch.add(space_to_depth(hr_warp1[i]), space_to_depth(hr_warp2[i]))
       
        z = torch.cat([v, k], 1)
       
        z = z.cuda()
        hr_curr = frnet(z)
        toimage(hr_curr,i,pathRes)  
    print('all images recostructed')
