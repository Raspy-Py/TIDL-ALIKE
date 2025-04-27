import logging
import torch
from alnet import ALNet

import os
import onnxruntime
import torch.nn as nn

class FeatureExtractor(ALNet):
    def __init__(self,
                 c1: int = 32, c2: int = 64, c3: int = 128, c4: int = 128, dim: int = 128,
                 single_head: bool = False,
                 device: str = 'cpu',
                 model_path: str = '',
                 manual_norm: bool = False,
                 manual_norm_eps: float = 1e-8):
        super().__init__(c1, c2, c3, c4, dim, single_head)

        self.device = device
        self.manual_norm = manual_norm # whether to normalize the descriptor manually
        self.manual_norm_eps = manual_norm_eps # epsilon for manual normalization

        if model_path != '':
            state_dict = torch.load(model_path, self.device)
            self.load_state_dict(state_dict)
            self.to(self.device)
            self.eval()
            logging.info(f'Loaded model parameters from {model_path}')
            logging.info(
                f"Number of model parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e3}KB")

    def extract_dense_map(self, image, ret_dict=False):
        scores_map, descriptor_map = super().forward(image)
        if self.manual_norm:
            norm = torch.sqrt(torch.sum(descriptor_map**2, dim=1, keepdim=True))
            #norm = 1 / (norm + self.manual_norm_eps)
            descriptor_map = descriptor_map / (norm + self.manual_norm_eps)
        else:
            descriptor_map = torch.nn.functional.normalize(descriptor_map, p=2, dim=1)

        return descriptor_map, scores_map

    def forward(self, image):
        with torch.no_grad():
            #descriptor_map, scores_map = self.extract_dense_map(image)
            scores_map, descriptor_map = super().forward(image)

        return descriptor_map, scores_map
    


class FeatureExtractorSingleOut(ALNet):
    def __init__(self,
                 c1: int = 32, c2: int = 64, c3: int = 128, c4: int = 128, dim: int = 128,
                 single_head: bool = False,
                 device: str = 'cpu',
                 model_path: str = '',
                 manual_norm: bool = False,
                 manual_norm_eps: float = 1e-8):
        super().__init__(c1, c2, c3, c4, dim, single_head)

        self.device = device
        self.manual_norm = manual_norm # whether to normalize the descriptor manually
        self.manual_norm_eps = manual_norm_eps # epsilon for manual normalization

        if model_path != '':
            state_dict = torch.load(model_path, self.device)
            self.load_state_dict(state_dict)
            self.to(self.device)
            self.eval()
            logging.info(f'Loaded model parameters from {model_path}')
            logging.info(
                f"Number of model parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e3}KB")

    def extract_dense_map(self, image, ret_dict=False):
        '''
        # TODO: Probably should be removed for efficiency
        # ====================================================
        # check image size, should be integer multiples of 2^5
        # if it is not a integer multiples of 2^5, padding zeros
        device = image.device
        c, h, w = image.shape
        h_ = math.ceil(h / 32) * 32 if h % 32 != 0 else h
        w_ = math.ceil(w / 32) * 32 if w % 32 != 0 else w
        if h_ != h:
            h_padding = torch.zeros(c, h_ - h, w, device=device)
            image = torch.cat([image, h_padding], dim=2)
        if w_ != w:
            w_padding = torch.zeros(c, h_, w_ - w, device=device)
            image = torch.cat([image, w_padding], dim=3)
        # ====================================================
        '''
        scores_map, descriptor_map = super().forward(image)
        '''
        # ====================================================
        if h_ != h or w_ != w:
            descriptor_map = descriptor_map[:, :h, :w]
            scores_map = scores_map[:, :h, :w]  # 1xHxW
        # ====================================================
        '''
        if self.manual_norm:
            norm = torch.sqrt(torch.sum(descriptor_map**2, dim=1, keepdim=True))
            #norm = 1 / (norm + self.manual_norm_eps)
            descriptor_map = descriptor_map / (norm + self.manual_norm_eps)
        else:
            descriptor_map = torch.nn.functional.normalize(descriptor_map, p=2, dim=1)

        return descriptor_map, scores_map

    def forward(self, image):
        with torch.no_grad():
            result = super().forward(image)
        return result
    

class FeatureExtractorORT(nn.Module):
    def __init__(self, model_path: str, artifacts_path: str = None, use_cuda: bool = False):
        
        self.use_cuda = use_cuda
        self.input_node = "image"
        self.artifacts_folder = artifacts_path
        self.model_path = model_path
        self.session = self.create_session()

    def create_session(self):
        if self.use_cuda:
            session = onnxruntime.InferenceSession(
                self.model_path,
                providers=["CUDAExecutionProvider"],
                sess_options=onnxruntime.SessionOptions(),
            )
        else:
            session = onnxruntime.InferenceSession(
                self.model_path,
                providers=["TIDLExecutionProvider"],
                provider_options=[self.get_tidl_inference_options()],
                sess_options=onnxruntime.SessionOptions(),
            )

        return session

    def get_tidl_inference_options(self):
        return {
            "tidl_tools_path": os.environ.get("TIDL_TOOLS_PATH", "/home/workdir/tidl_tools"),
            "artifacts_folder": self.artifacts_folder,
            "debug_level": 0,
        }
    
    def get_input_dict(self):    
        inputs_dict = {}
        inputs = self.session.get_outputs()
        for input in inputs:
            inputs_dict[input.name] = input.shape
        return inputs_dict

    def get_output_dict(self):
        ouputs_dict = {}
        outputs = self.session.get_outputs()
        for output in outputs:
            ouputs_dict[output.name] = output.shape
        return ouputs_dict
    
    def forward(self, image):
        descriptor_map, scores_map = self.session.run(None, {self.input_node: image})
        return descriptor_map, scores_map