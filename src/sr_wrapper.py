import yaml
import numpy as np
import torch

import utility
import model
import loss
from option import args
from data import common
global model

""" Implementation of APIs for super-resolution models

    Example usage (given a video file)
    =================================
    video = np.array(imageio.mimread(video_name))
    target = video[1, :, :]
    target_lr = np.resize(target, (lr_size, lr_size, 3))
    
    model = SuperResolutionModel("temp.yaml")
    prediction = predict_with_lr_video(target_lr)
"""
class SuperResolutionModel():
    def __init__(self, config_path, checkpoint='None'):
        super(SuperResolutionModel, self).__init__()
        torch.manual_seed(args.seed)
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # config parameters
        generator_params = config['model_params']['generator_params']
        self.shape = config['dataset_params']['frame_shape']
        self.use_lr_video = generator_params.get('use_lr_video', True)
        self.lr_size = generator_params.get('lr_size', 256)
        self.generator_type = generator_params.get('generator_type', 'edsr')
        self.n_feats = generator_params.get('n_feats', 256)
        self.scale = [int(self.shape[1] / self.lr_size)]
        self.device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # initialize weights
        if checkpoint == 'None':
            checkpoint = config['checkpoint_params']['checkpoint_path']

        args.pre_train = checkpoint
        args.scale = self.scale
        args.test_only = True
        args.n_feats = self.n_feats
        args.self_ensemble = True
        args.res_scale = 0.1
        args.n_resblocks = 32
        args.cpu = True #False if torch.cuda.is_available() else True

        checkpoint = utility.checkpoint(args)
        # configure modules
        self.generator = model.Model(args, checkpoint)
        self.generator.eval()

        self.args = args
        timing_enabled = True
        self.times = []

    def get_shape(self):
        return tuple(self.shape)


    def get_lr_video_info(self):
        return self.use_lr_video, self.lr_size


    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(self.device)

        return [_prepare(a) for a in args]


    def predict_with_lr_video(self, target_lr):
        """ predict and return the target RGB frame 
            from a low-res version of it. 
        """
        lr = target_lr
        lr, = common.set_channel(lr, n_channels=self.args.n_colors)
        lr, = common.np2Tensor(lr, rgb_range=self.args.rgb_range)
        lr, = self.prepare(lr.unsqueeze(0))
        sr = self.generator(lr, 0)
        sr = utility.quantize(sr, self.args.rgb_range).squeeze(0)

        normalized = sr * 255 / self.args.rgb_range
        ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
        return ndarr

