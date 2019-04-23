import os
import torch
from models.mobilenet import MobileNet


preinit="./init.pkl"
net=MobileNet()

if not os.path.axist(preinit):
  torch.save(net.state_dict(),preinit)
else:
  net.state_dict(torch.load(preinit))



