import torch
import torch.nn as nn
import warnings

class ECGNetworkWarning(UserWarning):
    pass

class spatialResidualBlock(nn.Module):
    def __init__(self,in_channels=(64,64),out_channels=(64,64),kernel_size=7,stride=1,groups=1,bias=True,padding=3,dropout=False):
        super(spatialResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels[0],
                               out_channels=out_channels[0],
                               kernel_size=(kernel_size,1),
                               stride=stride,
                               groups=groups,
                               bias=bias,
                               padding=(padding,0))
        self.batchNorm1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(in_channels=in_channels[1],
                               out_channels=out_channels[1],
                               kernel_size=(kernel_size,1),
                               stride=stride,
                               groups=groups,
                               bias=bias,
                               padding=(padding,0))
        self.batchNorm2 = nn.BatchNorm2d(out_channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = dropout
        self.drop = nn.Dropout()
        
        
        if in_channels[0] != out_channels[-1]:
            self.resampleInput = nn.Sequential(nn.Conv2d(in_channels=in_channels[0],
                                                         out_channels=out_channels[-1],
                                                         kernel_size=(1,1),
                                                         bias=bias,
                                                         padding = 0),
                                               nn.BatchNorm2d(out_channels[-1]))
        else:
            self.resampleInput = None

    def forward(self,X):
        if self.resampleInput is not None:
            identity = self.resampleInput(X)
        else:
            identity = X
        features = self.conv1(X)
        features = self.batchNorm1(features)
        features = self.relu(features)

        features = self.conv2(features)
        features = self.batchNorm2(features)
        if self.dropout:
            features = self.drop(features)
        features += identity
        features = self.relu(features)
        return features

class temporalResidualBlock(nn.Module):
    def __init__(self,in_channels=(64,64),out_channels=(64,64),kernel_size=3,stride=1,groups=1,bias=True,padding=1,dropout=False):
        super(temporalResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels[0],
                               out_channels=out_channels[0],
                               kernel_size=(1,kernel_size),
                               stride=stride,
                               groups=groups,
                               bias=bias,
                               padding=(0,padding))
        self.conv2 = nn.Conv2d(in_channels=in_channels[1],
                               out_channels=out_channels[1],
                               kernel_size=(1,kernel_size),
                               stride=stride,
                               groups=groups,
                               bias=bias,
                               padding=(0,padding))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = dropout
        self.drop = nn.Dropout()
        self.batchNorm1 = nn.BatchNorm2d(out_channels[0])
        self.batchNorm2 = nn.BatchNorm2d(out_channels[1])

        if in_channels[0] != out_channels[-1]:
            self.resampleInput = nn.Sequential(nn.Conv2d(in_channels=in_channels[0],
                                                         out_channels=out_channels[-1],
                                                         kernel_size=(1,1),
                                                         bias=bias,
                                                         padding = 0),
                                               nn.BatchNorm2d(out_channels[-1]))
        else:
            self.resampleInput = None

    def forward(self,X):
        if self.resampleInput is not None:
            identity = self.resampleInput(X)
        else:
            identity = X
        features = self.conv1(X)
        features = self.batchNorm1(features)
        features = self.relu(features)

        features = self.conv2(features)
        features = self.batchNorm2(features)
        if self.dropout:
            features = self.drop(features)
        
        features += identity
        features = self.relu(features)
        return features


class ECG_SpatioTemporalNet(torch.nn.Module):
    import torch as tch
    import torch.nn as nn
    def __init__(self,temporalResidualBlockParams,spatialResidualBlockParams,firstLayerParams,lastLayerParams,integrationMethod = 'add',problemType='Binary'):
        super(ECG_SpatioTemporalNet, self).__init__()
        self.firstLayer = nn.Sequential(nn.Conv2d(in_channels=firstLayerParams['in_channels'],
                                                  out_channels=firstLayerParams['out_channels'],
                                                  kernel_size=firstLayerParams['kernel_size'],
                                                  bias=firstLayerParams['bias'],
                                                  padding=(0,int(firstLayerParams['kernel_size'][1]/2))),
                                        nn.BatchNorm2d(firstLayerParams['out_channels']),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d((1,firstLayerParams['maxPoolKernel'])))
        self.residualBlocks_time = self._generateResidualBlocks(**temporalResidualBlockParams)

        self.residualBlocks_space = self._generateResidualBlocks(**spatialResidualBlockParams)
        if integrationMethod == 'add':
            integrationChannels = temporalResidualBlockParams['out_channels'][-1][-1]
        elif integrationMethod == 'concat':
            integrationChannels = temporalResidualBlockParams['out_channels'][-1][-1] + spatialResidualBlockParams['out_channels'][-1][-1]
        else:
            warnings.warn(f'Unknown concatenation method. Defaulting to addtion.',ECGNetworkWarning)
            integrationChannels = temporalResidualBlockParams['out_channels'][-1][-1]

        self.integrationBlock = nn.Sequential(nn.Conv2d(in_channels = integrationChannels,
                                                        out_channels = integrationChannels,
                                                        kernel_size = (3,3),
                                                        bias=firstLayerParams['bias'],
                                                        padding =1),
                                              nn.BatchNorm2d(integrationChannels),
                                              nn.Dropout(),
                                              nn.ReLU(inplace=True))

        self.finalLayer = nn.Sequential(nn.AdaptiveAvgPool2d(lastLayerParams['maxPoolSize']),
                                        nn.Flatten(),
                                        nn.Linear(in_features=lastLayerParams['maxPoolSize'][0]*lastLayerParams['maxPoolSize'][1]*integrationChannels,
                                                  out_features=1))
        self.integrationMethod = integrationMethod
        self.problemType = problemType
        if problemType == 'Binary':
            self.finalLayer = nn.Sequential(*self.finalLayer,nn.Sigmoid())
        elif problemType == 'Regression':
            self.finalLayer = nn.Sequential(*self.finalLayer,nn.ReLU())
        elif problemType == 'RegressionSoftPlus':
            self.finalLayer = nn.Sequential(*self.finalLayer,nn.Softplus())
        elif problemType == 'BCELogits':
            pass #exclude adding the relu or sigmoid
    def forward(self,X):
        resInputs = self.firstLayer(X)
        spatialFeatures = self.residualBlocks_space(resInputs)
        temporalFeatures = self.residualBlocks_time(resInputs)
        if self.integrationMethod == 'add':
            linearInputs = spatialFeatures + temporalFeatures
        elif self.integrationMethod == 'concat':
            linearInputs = torch.cat((spatialFeatures,temporalFeatures),dim=1)#concatenate the channels
        else:
            warnings.warn(f'Unknown concatenation method. Defaulting to addtion.',ECGNetworkWarning)
            linearInputs = spatialFeatures + temporalFeatures
        linearInputs = self.integrationBlock(linearInputs)
        output = self.finalLayer(linearInputs)
        return output


    def _generateResidualBlocks(self,numLayers,in_channels,out_channels,kernel_size,dropout,bias,padding,blockType):
        layerList = []
        for layerIx in range(numLayers):
            if blockType == 'Temporal':
                layerList.append(temporalResidualBlock(in_channels=in_channels[layerIx],
                                                   out_channels=out_channels[layerIx],
                                                   kernel_size=kernel_size[layerIx],
                                                   dropout=dropout[layerIx],
                                                   bias=bias,
                                                   padding=padding[layerIx]))
            if blockType == 'Spatial':
                layerList.append(spatialResidualBlock(in_channels=in_channels[layerIx],
                                                   out_channels=out_channels[layerIx],
                                                   kernel_size=kernel_size[layerIx],
                                                   dropout=dropout[layerIx],
                                                   bias=bias,
                                                   padding=padding[layerIx]))
        return nn.Sequential(*layerList)

class BaselineConvNet(nn.Module):

    def __init__(self, classification=False, avg_embeddings=False):
        super(BaselineConvNet, self).__init__()
        self.classification = classification
        self.avg_embeddings = avg_embeddings
        self.conv1 = nn.Conv1d(in_channels=1, 
                               out_channels=16, 
                               kernel_size=7, 
                               stride=4)
        self.batch_norm1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.Conv1d(in_channels=16,
                               out_channels=32,
                               kernel_size=7,
                               stride=3)
        self.batch_norm2 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(in_channels=32,
                               out_channels=64,
                               kernel_size=5,
                               stride=2)
        self.batch_norm3 = nn.BatchNorm1d(64)
        
        self.conv4 = nn.Conv1d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1)
        self.batch_norm4 = nn.BatchNorm1d(64)
        
        self.conv5 = nn.Conv1d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               stride=1)
        self.batch_norm5 = nn.BatchNorm1d(128)
        
        self.conv6 = nn.Conv1d(in_channels=128,
                               out_channels=256,
                               kernel_size=3,
                               stride=1)
        self.batch_norm6 = nn.BatchNorm1d(256)
        
        
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.finalLayer = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        ) if self.classification else nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, 1),

        )
    
    def forward(self, x):

        """ Forward Pass
        Args:
            x(torch.Tensor): Input tensor of shape (batch_size, 8, sequence_length)
        Returns:
            h(torch.Tensor): Output tensor of shape (batch_size, 1, 128) if avg_embeddings is True else (batch_size, 8, 128) and (batch_size, 1) if classification is True else (batch_size, 1) if classification
        """
        x = x.squeeze(1)  # Remove the channel dimension
        batch_size = x.shape[0]
        nviews = x.shape[1]
        if self.classification:
            self.avg_embeddings = True

        # h = X.view(-1, 1, samples)

        # x_i = self.batch_norm1(self.activation(self.conv1(h)))
        # x_i = self.batch_norm2(self.activation(self.conv2(x_i)))
        # x_i = self.batch_norm3(self.activation(self.conv3(x_i)))
        # x_i = self.batch_norm4(self.activation(self.conv4(x_i)))
        # x_i = self.batch_norm5(self.activation(self.conv5(x_i)))
        # x_i = self.batch_norm6(self.activation(self.conv6(x_i)))
        # x_i = self.avg_pool(x_i)
        # x_i = nn.Flatten()(x_i)
        # h = x_i.view(batch_size, nviews, -1)

        h = torch.empty(batch_size, nviews, 256, device=x.device)

        for i in range(nviews):
            x_i = x[:, i, :].unsqueeze(1)

            x_i = self.batch_norm1(self.activation(self.conv1(x_i)))
            
            x_i = self.batch_norm2(self.activation(self.conv2(x_i)))
            
            x_i = self.batch_norm3(self.activation(self.conv3(x_i)))
            
            x_i = self.batch_norm4(self.activation(self.conv4(x_i)))
            
            x_i = self.batch_norm5(self.activation(self.conv5(x_i)))
            
            x_i = self.batch_norm6(self.activation(self.conv6(x_i)))
            
            x_i = self.avg_pool(x_i)
            
            x_i = nn.Flatten()(x_i)
            

            h[:, i, :] = x_i


        if self.avg_embeddings:
            h = h.mean(dim=1, keepdim=True)
        

        h = self.finalLayer(h)
        

        if self.classification:
            h = h.squeeze(1)
            

        return h
