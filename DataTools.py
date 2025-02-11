import numpy as np
import torch
from torch.utils.data import Dataset
import os
import torch.nn.functional as F
import Loader
import torchvision.transforms as transforms

class DataLoaderError(Exception):
    pass

class dummyDataDataset(Dataset):
    def __init__(self, freqArray, ampArray, phaseArray, diseaseFlagArray, diseaseAmountArray, leads=8, timeSteps=5000):
        self.freqs = freqArray.tolist()
        self.amps = ampArray.tolist()
        self.phases = phaseArray.tolist()
        self.diseaseFlags = diseaseFlagArray.tolist()
        self.diseaseAmount = diseaseAmountArray.tolist()
        self.leads = leads
        self.timeSteps = timeSteps
        
    def __getitem__(self, item):
        data = self.generateSample(freqs=self.freqs[item],
                                   amps=self.amps[item],
                                   phases=self.phases[item],
                                   diseaseFlag=self.diseaseFlags[item],
                                   diseaseAmount=self.diseaseAmount[item])
        return torch.tensor(data).unsqueeze(0).float(),  torch.tensor(self.diseaseFlags[item]).float()
    
    def __len__(self):
        return len(self.diseaseFlags)
    
    def generateSample(self, freqs, amps, phases, diseaseFlag, diseaseAmount):
        dt = 10/self.timeSteps
        timeVec = np.linspace(0, 10-dt, self.timeSteps)
        data = np.tile((np.zeros(timeVec.shape)), (self.leads, 1))
        
        for lead in range(self.leads):
            for fr,am,ph in zip(freqs[lead], amps[lead], phases[lead]):
                data[lead, :] += np.sin(timeVec * np.pi * 2 * fr + ph) * am
            if diseaseFlag:
                diseaseSignature = np.zeros(data[lead, :].shape)
                diseaseSignature[0:round(self.timeSteps/5)] = np.sin(timeVec[0:round(self.timeSteps/5)] * np.pi * 2 * (1/4)) * diseaseAmount[lead]
                diseaseSignature[round(self.timeSteps/2):round(3*self.timeSteps/4)] = \
                    -np.sin(timeVec[round(self.timeSteps/2):round(3*self.timeSteps/4)] * np.pi * 2 * (1/5)) * diseaseAmount[lead]
                data[lead, :] += diseaseSignature
                
                
        return data
    

class ECG_KCL_Datasetloader(Dataset):
	def __init__(self,baseDir='',ecgs=[],kclVals=[],normalize =True, 
				 normMethod='0to1',rhythmType='Rhythm',allowMismatchTime=True,
				 mismatchFix='Pad',randomCrop=False,cropSize=2500,expectedTime=5000):
		self.baseDir = baseDir
		self.rhythmType = rhythmType
		self.normalize = normalize
		self.normMethod = normMethod
		self.ecgs = ecgs
		self.kclVals = kclVals
		self.expectedTime = expectedTime
		self.allowMismatchTime = allowMismatchTime
		self.mismatchFix = mismatchFix
		self.randomCrop = randomCrop
		self.cropSize = cropSize
		if self.randomCrop:
			self.expectedTime = self.cropSize

	def __getitem__(self,item):
		ecgName = self.ecgs[item].replace('.xml',f'_{self.rhythmType}.npy')
		ecgPath = os.path.join(self.baseDir,ecgName)
		ecgData = np.load(ecgPath)

		kclVal = torch.tensor(self.kclVals[item])
		ecgs = torch.tensor(ecgData).unsqueeze(0).float() #unsqueeze it to give it one channel\

		if self.randomCrop:
			startIx = 0
			if ecgs.shape[-1]-self.cropSize > 0:
				startIx = torch.randint(ecgs.shape[-1]-self.cropSize,(1,))
			ecgs = ecgs[...,startIx:startIx+self.cropSize]

		if ecgs.shape[-1] != self.expectedTime:
			if self.allowMismatchTime:
				if self.mismatchFix == 'Pad':
					ecgs=F.pad(ecgs,(0,self.expectedTime-ecgs.shape[-1]))
				if self.mismatchFix == 'Repeat':
					timeDiff = self.expectedTime - ecgs.shape[-1]
					ecgs=torch.cat((ecgs,ecgs[...,0:timeDiff]))

			else:
				raise DataLoaderError('You are not allowed to have mismatching data lengths.')

		if self.normalize:
			if self.normMethod == '0to1':
				if not torch.allclose(ecgs,torch.zeros_like(ecgs)):
					ecgs = ecgs - torch.min(ecgs)
					ecgs = ecgs / torch.max(ecgs)
				else:
					print(f'All zero data for item {item}, {ecgPath}')
			
		if torch.any(torch.isnan(ecgs)):
			print(f'Nans in the data for item {item}, {ecgPath}')
			raise DataLoaderError('Nans in data')
		return ecgs, kclVal

	def __len__(self):
		return len(self.ecgs)


class ECG_KCL_Augs_Datasetloader(Dataset):
	def __init__(self,baseDir='',ecgs=[],kclVals=[],normalize =True, 
				 normMethod='0to1',rhythmType='Rhythm',allowMismatchTime=True,
				 mismatchFix='Pad',randomCrop=False,cropSize=2500,expectedTime=5000):
		self.baseDir = baseDir
		self.rhythmType = rhythmType
		augmentation = [
			Loader.SpatialTransform(),
		]
		self.augs = Loader.TwoCropsTransform(transforms.Compose(augmentation))
		self.normalize = normalize
		self.normMethod = normMethod
		self.ecgs = ecgs
		self.kclVals = kclVals
		self.expectedTime = expectedTime
		self.allowMismatchTime = allowMismatchTime
		self.mismatchFix = mismatchFix
		self.randomCrop = randomCrop
		self.ecgs = [ecg for ecg in self.ecgs for _ in range(2)]
		self.kclVals = [kcl for kcl in self.kclVals for _ in range(2)]
		self.augmentationFlag = [False, True] * len(self.kclVals)
		self.cropSize = cropSize
		if self.randomCrop:
			self.expectedTime = self.cropSize
   		
	

	def __getitem__(self,item):
		ecgName = self.ecgs[item].replace('.xml',f'_{self.rhythmType}.npy')
		ecgPath = os.path.join(self.baseDir,ecgName)
		ecgData = np.load(ecgPath)

		kclVal = torch.tensor(self.kclVals[item])
		ecgs = torch.tensor(ecgData).unsqueeze(0).float() #unsqueeze it to give it one channel\

		if self.randomCrop:
			startIx = 0
			if ecgs.shape[-1]-self.cropSize > 0:
				startIx = torch.randint(ecgs.shape[-1]-self.cropSize,(1,))
			ecgs = ecgs[...,startIx:startIx+self.cropSize]

		if ecgs.shape[-1] != self.expectedTime:
			if self.allowMismatchTime:
				if self.mismatchFix == 'Pad':
					ecgs=F.pad(ecgs,(0,self.expectedTime-ecgs.shape[-1]))
				if self.mismatchFix == 'Repeat':
					timeDiff = self.expectedTime - ecgs.shape[-1]
					ecgs=torch.cat((ecgs,ecgs[...,0:timeDiff]))

			else:
				raise DataLoaderError('You are not allowed to have mismatching data lengths.')

		if self.normalize:
			if self.normMethod == '0to1':
				if not torch.allclose(ecgs,torch.zeros_like(ecgs)):
					ecgs = ecgs - torch.min(ecgs)
					ecgs = ecgs / torch.max(ecgs)
				else:
					print(f'All zero data for item {item}, {ecgPath}')
     
		if self.augmentationFlag[item]:
			ecgs = self.augs(ecgs)[0]
   
		if torch.any(torch.isnan(ecgs)):
			print(f'Nans in the data for item {item}, {ecgPath}')
			raise DataLoaderError('Nans in data')
		return ecgs, kclVal

	def __len__(self):
		return len(self.ecgs)