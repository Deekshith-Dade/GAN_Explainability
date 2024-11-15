import torch
import torch.nn as nn
import torch as tch
import torch.nn.functional as F

class coderBlock(nn.Module):
    def __init__(self,in_channels, out_channels, conv_kernel=(3,3,3),\
                 conv_stride=1, conv_pad=0,conv_dilation=1,conv_bias=True,\
                 do_max = True,max_kernel_size=(3,3,3),max_stride=1,max_pad=0,max_dilation=1,\
                 drop_prob=0.5,convType='standard',
                 do_drop=True,
                 actFun='relu',
                 do_batch_norm=True):
        super(coderBlock,self).__init__()
        layers = []
        self.convType = convType
        if convType == 'standard':
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=conv_kernel, stride=conv_stride, padding=conv_pad,
                                               dilation=conv_dilation, bias=conv_bias, padding_mode='zeros'))
        elif convType == 'transpose':
            layers.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=conv_kernel, stride=conv_stride, padding=conv_pad,
                                               dilation=conv_dilation, bias=conv_bias, padding_mode='zeros'))
        elif convType == 'none':
            pass
        else:
            warnings.warn('Invalid block type, skipping convolution layer',ECGNetworkWarning)
    
        if do_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if actFun == 'relu':
            layers.append(nn.ReLU())
        
        if do_max:
            layers.append(nn.MaxPool2d(kernel_size=max_kernel_size, stride=max_stride, padding=max_pad, dilation=max_dilation))
        if do_drop:
            layers.append(nn.Dropout(drop_prob))

        if len(layers) == 0:
            warnings.warn('No layers defined',SegmentationNetworkWarning)
        self.layers = nn.Sequential(*layers)

    def extra_repr(self):
        return f' {self.convType} covolution'

    def forward(self, x):
        return self.layers(x)

def define_default_generator_params():
    return dict(encoderParams=dict( convType=[['standard','standard','none']]*3,
                                    numStages=3,
                                    stagesPerLayer=[3,3,3],
                                    inCh=[[1,8,-1],[16,32,-1],[64,64,-1]],
                                    outCh=[[8,16,-1],[32,64,-1],[64,128,-1]],
                                    convKernel=[[(1,3),(1,3),(1,3)]]*3,
                                    conv_stride = [[1,1,0]]*3,
                                    conv_dilation = [[1,1,0]]*3,
                                    bias=[[True,True,False]]*3,
                                    convPad=[[(0,1),(0,1),(0,1)]]*3,
                                    dropProb=[[0.5, 0.5, 0.5]]*3,
                                    drop = [[True,True,False]]*3,
                                    batchNorm=[[True, True, False]]*3,
                                    actFun=[['relu','relu','none']]*3,
                                    maxPool=[[False,False,True]]*3,
                                    max_kernel_size=[[(),(),(1,3)]]*3,
                                    max_stride=[[(),(),(1,3)]]*3,
                                    max_padding=[[0,0,0]]*3,
                                    max_dilation=[[0,0,1]]*3,
                                    ),
                decoderParams=dict( convType=[['transpose','standard','standard']]*3,
                                    numStages=3,
                                    stagesPerLayer=[3,3,3],
                                    inCh=[[256,128,64],[128,64,32],[48,32,16]],
                                    outCh=[[128,64,64],[64,32,32],[32,16,16]],
                                    convKernel=[[(1,3),(1,3),(1,3)]]*3,
                                    conv_stride = [[(1,3),1,1]]*3,
                                    conv_dilation = [[1,1,1]]*3,
                                    bias=[[True,True,True]]*3,
                                    convPad=[[(0,0),(0,1),(0,2)]]*3,
                                    dropProb=[[0.5,0.5,0.5]]*3,
                                    drop = [[True,True,True]]*3,
                                    batchNorm=[[True,True,True]]*3,
                                    actFun=[['relu','relu','relu']]*3,
                                    maxPool=[[False,False,False]]*3,
                                    max_kernel_size=[[(),(),()]]*3,
                                    max_stride=[[0,0,3]]*3,
                                    max_padding=[[0,0,0]]*3,
                                    max_dilation=[[0,0,1]]*3, 
                                    ),
                latentParams= dict( inCh = 128,
                                    outCh = 128,
                                    convKernel = (1,3),
                                    conv_stride = 1,
                                    conv_dilation = 1,
                                    bias = True,
                                    convPad = (0,1),
                                    dropProb=0.5,
                                    drop = True,
                                    batchNorm=True,
                                    actFun='relu',
                                    maxPool=False,
                                    max_kernel_size=(),
                                    max_stride=[],
                                    max_padding=[],
                                    max_dilation=[]),
                finalParams = dict( inCh = 17,
                                    outCh = 1,
                                    convKernel = (1,3),
                                    conv_stride = 1,
                                    conv_dilation = 1,
                                    bias = True,
                                    convPad = (0,1),
                                    dropProb=0.5,
                                    drop = False,
                                    batchNorm=False,
                                    actFun='none',
                                    maxPool=False,
                                    max_kernel_size=(),
                                    max_stride=[],
                                    max_padding=[],
                                    max_dilation=[]))


class generator_V1(nn.Module):
    def __init__(self,encoderParams,decoderParams,latentParams,finalParams,verbose=True):
        super(generator_V1, self).__init__()
        self.verbose = verbose
        self._log("=======Generator Unet V1 Startup=======")
        self._generateBlock(encoderParams,blockType='encoder')
        self._generateBlock(decoderParams,blockType='decoder')
        self._generateBlock(latentParams,blockType='latent')
        self._generateBlock(finalParams,blockType='final')
        self._log("=======Startup Done=======")
        self._log(self)
        

    def _log(self,msg):
        if self.verbose:
            print(msg)
        
    def _generateBlock(self,params,blockType):
        self._log(f'Generating {blockType}')
        if blockType == 'final' or blockType == 'latent':
            block = coderBlock( in_channels =   params['inCh'], 
                                out_channels =  params['outCh'],
                                conv_kernel=    params['convKernel'],
                                conv_stride=    params['conv_stride'],
                                conv_pad=       params['convPad'],
                                conv_dilation=  params['conv_dilation'],
                                conv_bias=      params['bias'],
                                do_max =        params['maxPool'],
                                max_kernel_size=params['max_kernel_size'],
                                max_stride=     params['max_stride'],
                                max_pad=        params['max_padding'],
                                max_dilation=   params['max_dilation'],
                                drop_prob=      params['dropProb'],
                                convType=       'standard',
                                do_drop=        params['drop'],
                                actFun=         params['actFun'],
                                do_batch_norm=  params['batchNorm'])
            setattr(self,f'{blockType}Block',block)
            return
        stages = []
        for stage in range(params['numStages']):
            blocks = []
            for b in range(params['stagesPerLayer'][stage]):
                self._log(f'   ->Generating {b} of stage {stage}')
                block = coderBlock( in_channels =   params['inCh'][stage][b], 
                                    out_channels =  params['outCh'][stage][b],
                                    conv_kernel=    params['convKernel'][stage][b],
                                    conv_stride=    params['conv_stride'][stage][b],
                                    conv_pad=       params['convPad'][stage][b],
                                    conv_dilation=  params['conv_dilation'][stage][b],
                                    conv_bias=      params['bias'][stage][b],
                                    do_max =        params['maxPool'][stage][b],
                                    max_kernel_size=params['max_kernel_size'][stage][b],
                                    max_stride=     params['max_stride'][stage][b],
                                    max_pad=        params['max_padding'][stage][b],
                                    max_dilation=   params['max_dilation'][stage][b],
                                    drop_prob=      params['dropProb'][stage][b],
                                    convType=       params['convType'][stage][b],
                                    do_drop=        params['drop'][stage][b],
                                    actFun=         params['actFun'][stage][b],
                                    do_batch_norm=  params['batchNorm'][stage][b])
                blocks.append(block)
            stages.append(nn.Sequential(*blocks))
        setattr(self,f'{blockType}Stages',nn.Sequential(*stages))

        

    def encode(self,X):
        encoderOutputs = []
        self._log('Encoding')
        for encoderBlock in self.encoderStages:
            self._log(f'   ->Input: {X.shape}')
            X = encoderBlock(X)
            self._log(f'   ->Output: {X.shape}')
            encoderOutputs.append(X)
        return X, encoderOutputs

    def decode(self,X,encoderOutputs):
        self._log('Decoding')
        for decodeIx in range(len(self.encoderStages)):
            
            features = self.attachFeatures(X,encoderOutputs[-(decodeIx+1)])
            self._log(f'   ->in features: {features.shape}')
            X = self.decoderStages[decodeIx](features)
            self._log(f'   ->out features: {X.shape}')
        return X

    def attachFeatures(self,f1,f2):
        ##truncates m and/or n as needed to attach.
        self._log(f'        ->Attaching {[val for val in f1.shape]} to {[val for val in f2.shape]}')
        if f1.shape == f2.shape:
            return tch.cat((f1,f2),dim=1)

        minDims =  tch.min(tch.tensor([list(f1.shape),list(f2.shape)]),dim=0)[0]
        if len(minDims) == 4:#batch, channel, m, n
            return tch.cat((f1[:,:,:minDims[2],:minDims[3]],
                            f2[:,:,:minDims[2],:minDims[3]]),dim=1)
        if len(minDims) == 3:#channel, m, n
            return tch.cat((f1[:,:minDims[1],:minDims[2]],
                            f2[:,:minDims[1],:minDims[2]]),dim=1)



        

    def forward(self,InputX):
        X, encoderOutputs = self.encode(InputX)
        self._log(f'Latent Input: {X.shape}')
        X = self.latentBlock(X)
        self._log(f'Latent Output: {X.shape}')
        X = self.decode(X, encoderOutputs)
        X = self.attachFeatures(X,InputX)
        X = self.finalBlock(X)
        return X


def modifyData_addition(data, generatorOutput):
    return data + generatorOutput

def modificationPenalty(modifications, method, weights, dt=0.002):
    penalty = torch.tensor([0.0]).to(modifications.get_device())
    for m, w in zip(modifications, weights):
        if method == 'L1':
            penalty +=modifications.norm(1,dim=3).mean() * w
        if method == 'L1_dt':
            penalty +=tch.gradient(modifications,dim=3,spacing=dt)[0].norm(1,dim=3).mean() * w
    return penalty

    

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1,3), stride=1, padding=0):
        super(conv_block, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.layers(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel=(1,3), conv_stride=(1,1), 
                 conv_pad=0):
        super(EncoderBlock, self).__init__()
        self.conv_block = conv_block(in_channels, out_channels, kernel_size=conv_kernel, stride=conv_stride, padding=conv_pad)
        self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,3))
    
    def forward(self, x):
        x = self.conv_block(x)
        d = self.pool(x)
        return x, d

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1,3), stride=(1,3), padding=0):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=kernel_size, stride=stride, padding=(0,0))
        self.conv_block = conv_block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=(0,2))
    
    def forward(self, x, skip):
        x = self.upconv(x)
        x = self.concat_features(x, skip)
        x = self.conv_block(x)
        return x
    
    def concat_features(self, f1, f2):
        
        if f1.shape == f2.shape:
            return tch.cat((f1,f2),dim=1)

        diffY = f2.size()[2] - f1.size()[2]
        diffX = f2.size()[3] - f1.size()[3]

        x1 = F.pad(f1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([f2, x1], dim=1)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.Encoder = nn.Sequential(
            EncoderBlock(1, 4),
            EncoderBlock(4, 8),
            EncoderBlock(8, 16),
            EncoderBlock(16, 32),
        )
        
        self.Latent = conv_block(32, 64, (1,3), stride=(1, 1), padding=(0, 1))
        
        self.Decoder = nn.Sequential(
            DecoderBlock(64, 32),
            DecoderBlock(32, 16),
            DecoderBlock(16, 8),
            DecoderBlock(8, 4),
        )
        
        self.final_block = torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=(1,3), padding=(0,1), stride=(1,1))
        
    def forward(self, x):
        x_og = x.clone()
        skips = []
        for encoder in self.Encoder:
            skip, x = encoder(x)
            skips.append(skip)
        
        x = self.Latent(x)
        for decoder, skip in zip(self.Decoder, reversed(skips)):
            x = decoder(x, skip)
        
        x = torch.cat([x, x_og], dim=1)
        return self.final_block(x)

        
        
        
        
        