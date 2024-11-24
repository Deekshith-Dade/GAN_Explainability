import pandas as pd
import numpy as np
import random

import torch
from torch import optim
import DataTools as DD
from generator import Generator, modificationPenalty, modifyData_addition
import network_params as netParams
from discriminator import ECG_SpatioTemporalNet
import training as T

import wandb
import datetime
import os
import pdb

baseDir = '/usr/sci/cibc/ProjectsAndScratch/DeekshithMLECG/explainability'
dataDir = '/usr/sci/cibc/Maprodxn/ClinicalECGData/AllClinicalECGs/'

# os.environ["WANDB_API_KEY"] = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpuIds = range(torch.cuda.device_count())
multipleGPUs = 1 if len(gpuIds) > 1 else 0

randSeed = 7777
logtowandb = True
modelSaveDir = baseDir + '/models/Explainability/'


optimType = 'adam'
numEpoch = 50


evalUpdates = range(0, numEpoch, 1)
batch_per_gpu = 80
batch_size = len(gpuIds) * batch_per_gpu
np.random.seed(randSeed)

timeCutoff = 900 #seconds
lowerCutoff = 0 #seconds

kclCohort = np.load(dataDir+'kclCohort_v1.npy',allow_pickle=True)
data_types = {
    'DeltaTime': float,   
    'KCLVal': float,    
    'ECGFile': str,     
    'PatId': int,       
    'KCLTest': str      
}
kclCohort = pd.DataFrame(kclCohort,columns=['DeltaTime','KCLVal','ECGFile','PatId','KCLTest']) 
for key in data_types.keys():
	kclCohort[key] = kclCohort[key].astype(data_types[key])

kclCohort = kclCohort[kclCohort['DeltaTime']<=timeCutoff]
kclCohort = kclCohort[kclCohort['DeltaTime']>lowerCutoff]

kclCohort = kclCohort.dropna(subset=['DeltaTime']) 
kclCohort = kclCohort.dropna(subset=['KCLVal']) 

ix = kclCohort.groupby('ECGFile')['DeltaTime'].idxmin()
kclCohort = kclCohort.loc[ix]

numECGs = len(kclCohort)
numPatients = len(np.unique(kclCohort['PatId']))

print('setting up train/val split')
numTest = int(0.1 * numPatients)
numTrain = numPatients - numTest
assert (numPatients == numTrain + numTest), "Train/Test spilt incorrectly"
RandomSeedSoAlswaysGetSameDatabseSplit = 1
patientIds = list(np.unique(kclCohort['PatId']))
random.Random(RandomSeedSoAlswaysGetSameDatabseSplit).shuffle(patientIds)

trainPatientInds = patientIds[:numTrain]
testPatientInds = patientIds[numTrain:numTest + numTrain]
trainECGs = kclCohort[kclCohort['PatId'].isin(trainPatientInds)]
testECGs = kclCohort[kclCohort['PatId'].isin(testPatientInds)]

desiredTrainingAmount = len(trainECGs)

if desiredTrainingAmount != 'all':
	if len(trainECGs)>desiredTrainingAmount:
		trainECGs = trainECGs.sample(n=desiredTrainingAmount)

kclTaskParams = dict(highThresh = 5, lowThresh=4, highThreshRestrict=8.5)
trainECGs_normal = trainECGs[(trainECGs['KCLVal']>=kclTaskParams['lowThresh']) & (trainECGs['KCLVal']<=kclTaskParams['highThresh'])]
trainECGs_abnormal = trainECGs[(trainECGs['KCLVal']>kclTaskParams['highThresh']) & (trainECGs['KCLVal']<=kclTaskParams['highThreshRestrict'])]


testECGs_normal = testECGs[(testECGs['KCLVal']>=kclTaskParams['lowThresh']) & (testECGs['KCLVal']<=kclTaskParams['highThresh'])]
testECGs_abnormal = testECGs[(testECGs['KCLVal']>kclTaskParams['highThresh']) & (testECGs['KCLVal']<=kclTaskParams['highThreshRestrict'])]

dataset_augs = DD.ECG_KCL_Augs_Datasetloader
dataset_regular = DD.ECG_KCL_Datasetloader

trainNormalDataset = dataset_regular(
    baseDir= dataDir + 'pythonData/',
    ecgs=trainECGs_normal['ECGFile'].tolist(),
    kclVals=trainECGs_normal['KCLVal'].tolist(),
    normalize=False,
    allowMismatchTime=False,
    randomCrop=True,
)

trainAbnormalDataset = dataset_augs(
    baseDir= dataDir + 'pythonData/',
    ecgs=trainECGs_abnormal['ECGFile'].tolist(),
    kclVals=trainECGs_abnormal['KCLVal'].tolist(),
    normalize=False,
    allowMismatchTime=False,
    randomCrop=True,
    
)

testNormalData = dataset_regular(
    baseDir= dataDir + 'pythonData/',
    ecgs=testECGs_normal['ECGFile'].tolist(),
    kclVals=testECGs_normal['KCLVal'].tolist(),
    normalize=False,
    allowMismatchTime=False,
    randomCrop=True,
)

testAbnormalData = dataset_regular(
    baseDir= dataDir + 'pythonData/',
    ecgs=testECGs_abnormal['ECGFile'].tolist(),
    kclVals=testECGs_abnormal['KCLVal'].tolist(),
    normalize=False,
    allowMismatchTime=False,
    randomCrop=True,
    
)

# Save the ECGs from testAbnormalData as a numpy array along with the KCL values
# ECGs = torch.empty((0, 8, 2500))
# KCLs = torch.empty(0)
# for i in range(len(testAbnormalData)):
#     ECGs = torch.cat((ECGs, testAbnormalData[i][0]), 0)
#     KCLs = torch.cat((KCLs, torch.tensor(testAbnormalData[i][1]).unsqueeze(0)), 0)

# file = dict(ECGs=ECGs, KCLs=KCLs)
# torch.save(file, f'{baseDir}/testAbnormalData/testAbnormalData.pt')

# pdb.set_trace()

trainNormalDataLoader = torch.utils.data.DataLoader(trainNormalDataset, shuffle=True, batch_size=batch_size, num_workers=32, drop_last=True)
trainAbnormalDataLoader = torch.utils.data.DataLoader(trainAbnormalDataset, shuffle=True, batch_size=batch_size, num_workers=32, drop_last=True)

testNormalDataLoader = torch.utils.data.DataLoader(testNormalData, shuffle=False, batch_size=batch_size, num_workers=32, drop_last=True)
testAbnormalDataLoader = torch.utils.data.DataLoader(testAbnormalData, shuffle=False, batch_size=batch_size, num_workers=32, drop_last=True)


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
 
# classificationIter_perEp = 5 * [0] + [0] * 10
discrimIter_perEp = numEpoch * [1]
genIter_perEp = [10] * numEpoch

modPenalties = dict(method=['l1','l1_dt'],weights=[1e-3,1e-3])  #  dict(method=['l2'],weights=[1e-5]) 
modificationPenaltyL = lambda m : modificationPenalty(m,method=modPenalties['method'],weights=modPenalties['weights'])
networkLabel = 'KCL_Explainability'
# LOSS PARAMS
lossParams = dict(learningRate_gen = 1e-4,learningRate_dis=1e-4,type = 'wgan',weights=dict(discriminatorWeight=1.,
																 descrimFakeWeight = 1.,
																 descrimTrueWeight = 1.,
																 classificationWeight=1.,
																 generatorWeight=1.,
																 gradWeight=1.,
																 modificationWeight=5.))
# Gen 1e-3, Dis 1e-4
if __name__ == "__main__":
    
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print("Main")
    # seed_everything(42)
    config = dict (
			  learning_rate = [lossParams['learningRate_gen'],lossParams['learningRate_dis']],
			  lossType = lossParams['type'],
			  loassParams = lossParams,
			  batchSize = batch_size,
			  architecture = f"{networkLabel}",
			  dataset_id = "KCL",
			  optimType = optimType,
			  randSeed= randSeed,
			  genIter_perEp = genIter_perEp,
			  discrimIter_perEp = discrimIter_perEp,
			#   classificationIter_perEp = classificationIter_perEp,
			  modPenalties=modPenalties,
            trainNormalSize = len(trainNormalDataset),
            trainAbnormalSize = len(trainAbnormalDataset),
			)
    if logtowandb:
        wandbrun = wandb.init(
		  project="KCLExplainability3",
		  notes=f"train_{'networkLabel'}, not using sigmoid outputs any more",
		  tags=["training","KCL"],
		  config=config,
		  entity="deekshith",
		  reinit=True,
		  name=f"{networkLabel}_{datetime.datetime.now()}",
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
                                #   classificationIter_perEp=classificationIter_perEp,
                                  evalUpdates=evalUpdates,
                                  lossFun=T.loss_wgan,
                                  logtowandb=logtowandb)