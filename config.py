from easydict import EasyDict

configer = EasyDict()


configer.dsize = (64, 64)
configer.n_channels = 23
configer.n_class = 63


configer.splitmode = 'split_{}x{}_2'.format(configer.dsize[0], configer.dsize[1])
#configer.modelbase = 'recognize_mobilefacenet'
configer.modelbase ='recognize_mobilenet'
#configer.modelbase = 'recognize_shufflev2'

configer.datatype = 'Multi'
if configer.datatype == 'Multi':
    #configer.usedChannels =[770,850,730,810,890]
    #configer.usedChannels =[770,850,730,810,890,750,790]
    #configer.usedChannels =[770,850,730]
    #configer.usedChannels = range(550,1000)[::20]
    #configer.usedChannels =[770,850,730,810,890,750,790,830,870,910,930,950,710,990,970,690,670,650,590,630,570]
    configer.usedChannels =[870]
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
configer.logspath = '/home/siminzhu/rewrite_recognize/logs/Mobilenet'
configer.mdlspath = '/home/siminzhu/rewrite_recognize/modelfiles/recognize/Mobilenet'


## training step
configer.batchsize = 64
configer.batchsize_test=64
configer.n_epoch   = 300

## learing rate
configer.lrbase = 0.005
configer.stepsize = 200
configer.gamma = 0.1

configer.cuda = True
