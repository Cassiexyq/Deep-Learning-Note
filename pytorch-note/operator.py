# -*- coding: utf-8 -*-

# @Author: xyq

import torch
import numpy as np
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1) # 直接tensor里 加1
print(a)

c = np.ones(5)
print(c)
d = torch.from_numpy(c)
np.add(c,1)
print(c)
print(d)

x = torch.rand(4,4)
y = x.view(16)
z = x.view(-1,8)
print(x.size(),y.size(),z.size())

if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x,device=device)
    x = x.to(device)
    z = x+y
    print(z)
    print(z.to("cpu",torch.double))