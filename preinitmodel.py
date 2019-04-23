import os
import torch
from models.mobilenet import MobileNet
from config import configer

preinit="./init.pkl"
net=MobileNet()
if configer.cuda and is_available(): net.cuda()

if not os.path.exists(preinit):
  torch.save(net.state_dict(),preinit)
else:
  net.state_dict(torch.load(preinit))



