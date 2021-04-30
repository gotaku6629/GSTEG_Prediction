"""
program test
"""

import torch
import numpy as np

a = torch.zeros([6, 6])
b = torch.ones([6, 6])
print('a.shape={}'.format(a.shape))
print('b.shape={}'.format(b.shape))

# (2,2)のテンソルを4つ加えて, (4,2,2)にするぞ!!
# unsqueezeが次元を上げてくれる!!
c = torch.cat([torch.randn(2,2).unsqueeze(0) for _ in range(4)], dim =0)
print('c.shape={}'.format(c.shape)) # torch.Size([4, 2, 2]) 

# テスト
rgb_feat = torch.randn(3,2,3)
print('rgb_feat={}'.format(rgb_feat))

for i in range(rgb_feat.shape[0]):
    print('rgb_feat[', i, ']={}'.format(rgb_feat[i]))
    #tensor([[ 0.0924, -1.3739,  1.4417],  # (2,3)
    #        [ 0.7881,  1.3122, -0.9455]])
    print('rgb_feat[', i, '].unsqueeze(0)={}'.format(rgb_feat[i].unsqueeze(0)))
    #tensor([[[ 0.0924, -1.3739,  1.4417], # (1, 2, 3)
    #         [ 0.7881,  1.3122, -0.9455]]])

s = torch.cat([rgb_feat[i].unsqueeze(0) for i in range(3)])
print('s={}'.format(s))

# テスト2
rgb_feat = torch.randn(3,2,3)
print('rgb_feat={}'.format(rgb_feat))
ss = torch.cat([rgb_feat[i].unsqueeze(0) for i in range(2)])
print('ss={}'.format(ss))
ss_2 = rgb_feat[2].unsqueeze(0)
print('ss_2={}'.format(ss_2))
ss = torch.cat([ss, ss_2], dim=0)
print('ss={}'.format(ss))  # OK


# テスト3
hwc_img = torch.rand(25,3,3,10,224,224)
print('hwc_img.shape={}'.format(hwc_img.shape))
whc_img = hwc_img.transpose(0,1)
print('whc_img.shape={}'.format(whc_img.shape))