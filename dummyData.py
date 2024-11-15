import sys, os, wandb, datetime
import numpy as np
import torch
from torch import optim
import pdb
import DataTools as DD

from generator import Generator, modificationPenalty, modifyData_addition
import network_params as netParams
from discriminator import ECG_SpatioTemporalNet, BaselineConvNet
import training as T

baseDir = '/usr/sci/cibc/ProjectsAndScratch/DeekshithMLECG/explainability'

os.environ["WANDB_API_KEY"] = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpuIds = range(torch.cuda.device_count())
multipleGPUs = 1 if len(gpuIds) > 1 else 0

randSeed = 7777
logtowandb = True
modelSaveDir = baseDir + 'models/Explainability_Dummy/'


optimType = 'adam'
numEpoch = 25

classificationIter_perEp = 5 * [0] + [0] * 10
discrimIter_perEp = numEpoch * [1]
genIter_perEp = [5] * numEpoch

evalUpdates = range(0, numEpoch, 1)
batch_per_gpu = 40
batch_size = len(gpuIds) * batch_per_gpu
np.random.seed(randSeed)


# Setting up dataset
def getDummyDataset(numData, percentDisease):
  numData = numData
  numLeads = 8
  percentTrain = 1.0
  percentDisease = percentDisease
  freqs = np.tile(np.array([2.,5.,10.]), (numData, numLeads, 1))
  freqs += np.random.rand(*freqs.shape)*2. -1.

  amps = np.tile(np.array([1., 1., .5]), (numData, numLeads, 1))
  amps += np.random.rand(*amps.shape)*.5 - .25

  phases = np.tile(np.array([0.1, 0.2, -0.1]), (numData, numLeads, 1))
  phases += np.random.rand(*phases.shape)*np.pi*.5

  diseaseFlag = np.random.rand(numData) < percentDisease

  diseaseMin = 3
  diseaseMax = 7
  diseaseAmount = (np.random.rand(numData, numLeads)*(diseaseMax-diseaseMin) + diseaseMin)
  diseaseAmount[~diseaseFlag, :] = 0
  
  Dataset = DD.dummyDataDataset(freqs[:int(numData*percentTrain),...],
                                   amps[:int(numData*percentTrain),...],
                                   phases[:int(numData*percentTrain),...],
                                   diseaseFlag[:int(numData*percentTrain)],
                                   diseaseAmount[:int(numData*percentTrain),...],
                                   leads=numLeads)
  
  return Dataset
  
numData = 100000
trainNormalDataset = getDummyDataset(numData, 0.0)
trainAbnormalDataset = getDummyDataset(numData, 1.0)

testNormalData = getDummyDataset(int(numData * 0.1), 0.0)
testAbnormalData = getDummyDataset(int(numData * 0.1), 1.0)

trainNormalDataLoader = torch.utils.data.DataLoader(trainNormalDataset, shuffle=True, batch_size=batch_size, num_workers=16, drop_last=True)
trainAbnormalDataLoader = torch.utils.data.DataLoader(trainAbnormalDataset, shuffle=True, batch_size=batch_size, num_workers=16, drop_last=True)

testNormalDataLoader = torch.utils.data.DataLoader(testNormalData, shuffle=False, batch_size=batch_size, num_workers=16, drop_last=True)
testAbnormalDataLoader = torch.utils.data.DataLoader(testAbnormalData, shuffle=False, batch_size=batch_size, num_workers=16, drop_last=True)




# GENERATOR
print("Setting up GENERATOR")
generator = Generator()
generator = generator.to(device)
if multipleGPUs:
	generator = torch.nn.DataParallel(generator, device_ids=gpuIds)

# DISCRIMINATOR
print("Setting up DISCRIMINATOR")
firstLayerParams = dict(in_channels=1,out_channels=32,bias=True,kernel_size=(1,7),maxPoolKernel=7)
lastLayerParams = dict(maxPoolSize=(8,1))
discriminatorParams =  {'temporalResidualBlockParams':netParams.spatioTemporalParams_v4['temporalResidualBlockParams'],
				  'spatialResidualBlockParams':netParams.spatioTemporalParams_v4['spatialResidualBlockParams'],
				  'integrationMethod':'concat','problemType':'BCELogits','firstLayerParams':firstLayerParams,'lastLayerParams':lastLayerParams}

discriminator = ECG_SpatioTemporalNet(**discriminatorParams) #BaselineConvNet(avg_embeddings=True)
discriminator = discriminator.to(device)
if multipleGPUs:
	discriminator = torch.nn.DataParallel(discriminator, device_ids=gpuIds)

modPenalties = dict(method=['l1','l1_dt'],weights=[1e-3,1e-3])
modificationPenaltyL = lambda m : modificationPenalty(m,method=modPenalties['method'],weights=modPenalties['weights'])

# LOSS PARAMS
lossParams = dict(learningRate_gen = 1e-3,learningRate_dis=1e-4,type = 'wgan',weights=dict(discriminatorWeight=1.,
																 descrimFakeWeight = 1.,
																 descrimTrueWeight = 1.,
																 classificationWeight=1.,
																 generatorWeight=1.,
																 gradWeight=1.,
																 modificationWeight=1.))

if __name__ == "__main__":
    print("Main")
    
    config = dict (
			  learning_rate = [lossParams['learningRate_gen'],lossParams['learningRate_dis']],
			  lossType = lossParams['type'],
			  loassParams = lossParams,
			  batchSize = batch_size,
			  architecture = f"{'networkLabel'}",
			  dataset_id = "SynthSins",
			  optimType = optimType,
			  randSeed= randSeed,
			  genIter_perEp = genIter_perEp,
			  discrimIter_perEp = discrimIter_perEp,
			  classificationIter_perEp = classificationIter_perEp,
			  modPenalties=modPenalties,
			)
    if logtowandb:
        wandbrun = wandb.init(
		  project="TestExplainability",
		  notes=f"train_{'networkLabel'}, not using sigmoid outputs any more",
		  tags=["training","dummyData_v1"],
		  config=config,
		  entity="deekshith",
		  reinit=True,
		  name=f"{'networkLabel'}_{datetime.datetime.now()}",
		)
    
    torch.cuda.empty_cache()
    
    if optimType == 'adam':
        optim_generator = optim.Adam(generator.parameters(), lr=lossParams['learningRate_gen'], betas=(0.0, 0.9))
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=lossParams['learningRate_dis'], betas=(0.0, 0.9))
    
    print("Starting Training")
    T.trainExplainabilityNetworks(discriminator=discriminator,
                                  generator=generator,
                                  optim_discriminator=optim_discriminator,
                                  optim_generator=optim_generator,
                                  modificationFunction=modifyData_addition,
                                  modificationPenalty=modificationPenaltyL,
                                  trainNormalData=trainNormalDataLoader,
                                  trainAbnormalData=trainAbnormalDataLoader,
                                  testNormalData  = testNormalDataLoader,
                                  testAbnormalData = testAbnormalDataLoader,
                                  epochs=numEpoch,
                                  lossParams=lossParams,
                                  modelSaveDir=modelSaveDir,
                                  label='networkLabel',
                                  genIter_perEp=genIter_perEp,
                                  discrimIter_perEp=discrimIter_perEp,
                                  classificationIter_perEp=classificationIter_perEp,
                                  evalUpdates=evalUpdates,
                                  lossFun=T.loss_wgan,
                                  logtowandb=logtowandb)



                                  