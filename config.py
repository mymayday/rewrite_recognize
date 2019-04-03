from easydict import EasyDict

configer = EasyDict()


configer.dsize = (64, 64)
configer.n_channels = 23
configer.n_class = 63


configer.splitmode = 'split_{}x{}_1'.format(configer.dsize[0], configer.dsize[1])
configer.modelbase = 'recognize_mobilefacenet'


configer.datatype = 'Multi'
if configer.datatype == 'Multi':
    configer.usedChannels = [550]
    configer.n_usedChannels = len(configer.usedChannels)
    configer.modelname = '{}_{}_{}chs_{}sta_20nm'.\
                    format(configer.modelbase, configer.splitmode, configer.n_usedChannels, configer.usedChannels[0])
elif configer.datatype == 'RGB':
    configer.usedChannels = 'RGB'
    configer.n_usedChannels = len(configer.usedChannels)
    configer.modelname = '{}_{}_{}'.\
                    format(configer.modelbase, configer.splitmode, configer.usedChannels)


configer.datapath = '/datasets/ECUST2019_{}x{}'.\
                                format(configer.dsize[0], configer.dsize[1])
configer.logspath = '/home/siminzhu/rewrite_recognize/logs'
configer.mdlspath = '/home/siminzhu/rewrite_recognize/modelfiles/recognize'


## training step
configer.batchsize = 64
configer.n_epoch   = 200

## learing rate
configer.lrbase = 1e-4
configer.stepsize = 200
configer.gamma = 0.1

configer.cuda = True
