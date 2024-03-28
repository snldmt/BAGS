# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""

import numpy as np
import os
# from typing import Dict, Literal, Optional, Tuple

import torch
from torch import nn

class BlurKernel(nn.Module):
    def __init__(self, num_img, H=400, W=600, img_embed=32, ks=17):
        super().__init__()
        self.num_img = num_img
        self.W, self.H = W, H

        self.img_embed_cnl = img_embed

        self.min_freq, self.max_freq, self.num_frequencies = 0.0, 3.0, 4

        self.embedding_camera = nn.Embedding(self.num_img, self.img_embed_cnl)

        self.mlp_base_mlp = torch.nn.Sequential(
            torch.nn.Conv2d(48+32, 64, 1, bias=False), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 1, bias=False), torch.nn.ReLU(),
            )

        print('this is single res kernel', ks)
        
        self.mlp_head1 = torch.nn.Conv2d(64, ks**2, 1, bias=False)
        self.mlp_mask1 = torch.nn.Conv2d(64, 1, 1, bias=False)

        self.conv_rgb = torch.nn.Sequential(
            torch.nn.Conv2d(4, 64, 5,padding=2), torch.nn.ReLU(), torch.nn.InstanceNorm2d(64),
            torch.nn.Conv2d(64, 64, 5,padding=2), torch.nn.ReLU(), torch.nn.InstanceNorm2d(64),
            torch.nn.Conv2d(64, 32, 3,padding=1)
            )

    def forward(self, img_idx, pos_enc, img, iter):
        rgbd_feat = self.conv_rgb(img)

        img_embed = self.embedding_camera(torch.LongTensor([img_idx]).cuda())[None, None]
        img_embed = img_embed.expand(pos_enc.shape[0],pos_enc.shape[1],pos_enc.shape[2],img_embed.shape[-1])
        inp = torch.cat([img_embed,pos_enc],-1).permute(0,3,1,2)

        feat = self.mlp_base_mlp(torch.cat([inp,rgbd_feat],1))

        weight = self.mlp_head1(feat)
        mask = self.mlp_mask1(feat)

        weight = torch.softmax(weight, dim=1)
        mask = torch.sigmoid(mask)

        return weight, mask



