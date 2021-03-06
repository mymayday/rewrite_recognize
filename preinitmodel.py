import os
import torch
from models.mobilenet import MobileNet
from config import configer
from torch.cuda import is_available

preinit="./init.pkl"
net=MobileNet()
if configer.cuda and is_available(): net.cuda()

if not os.path.exists(preinit):
  torch.save(net.state_dict(),preinit)
else:
  net.load_state_dict(torch.load(preinit))



