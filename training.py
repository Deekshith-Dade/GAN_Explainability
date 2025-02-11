import torch
import matplotlib.pyplot as plt
import wandb
from itertools import cycle
import sys
import datetime
import os
import numpy as np
import json
from scipy.io import savemat

torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpuIds = range(torch.cuda.device_count())
multipleGPUs = 1 if len(gpuIds) > 1 else 0

def loss_wgan(discriminator, trueResults, fakeResults, trueData, fakeData, weights):
    compareSize = [trueData.shape[0]] + [1] * (len(trueData.shape) - 1)
    eps = torch.rand(compareSize).to(device)
    eps = eps.expand_as(trueData)
    with torch.enable_grad():
        interpValue = eps * trueData + (1.0 - eps) * fakeData
        interpValue.requires_grad_()
        
        interp_prediction = discriminator(interpValue)
        grads = torch.autograd.grad(outputs=interp_prediction, inputs=interpValue, grad_outputs=torch.ones_like(interp_prediction),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradPenalty = torch.pow(grads.norm(2, dim=1)-1, 2).mean()
    
    descrimPenalty = -(trueResults.mean() * weights['descrimTrueWeight']) + (fakeResults.mean() * weights['descrimFakeWeight'])
    wganloss = descrimPenalty*weights['discriminatorWeight'] + gradPenalty*weights['gradWeight']
    # Add drift penalty
    # wganloss += ((torch.mean(trueResults**2) + torch.mean(fakeResults**2)) / 2) * 1e-4
    
    return wganloss

def eval_generator_discriminator(discriminator, generator, testNormalData, testAbnormalData, lossFun, lossParams, modificationFunction, modificationPenalty, directory, epoch):
    discriminator.eval()
    generator.eval()
    plt.figure(1)  
    subDims = [8, 4]
    fig, axes = plt.subplots(subDims[0], subDims[1])
    figNeeded = 1
    diseaseFound = 0
    healthFound = 0
    with torch.no_grad():
        runningLoss_wgan = 0.0
        runningLoss_generator = 0.0
        
        normalData_iter = iter(testNormalData)
        abnormalData_iter = iter(testAbnormalData)
        
        for ix in range(min(len(testNormalData), len(testAbnormalData))):
            print(f'Batch {ix} of {len(testAbnormalData)}', end='\r')
            normalData, flag = next(normalData_iter)
            normalData = normalData.to(device)
            
            abnormalData, kclVals = next(abnormalData_iter)
            abnormalData = abnormalData.to(device)
            
            
            TrueResults = discriminator(normalData)
            
            modifications = generator(abnormalData)
            fakeData = modificationFunction(abnormalData, modifications)
            
            FakeResults = discriminator(fakeData)
            wganLoss = lossFun(discriminator,TrueResults,FakeResults,normalData,abnormalData,lossParams['weights'])
            
            # modifications = generator(trueData[flag==1,...])
            # fakeData = modificationFunction(trueData[flag==1,...], modifications)
            generatorLoss = (-FakeResults).mean()*lossParams['weights']['generatorWeight'] +\
							modificationPenalty(modifications)*lossParams['weights']['modificationWeight']
            runningLoss_wgan += wganLoss.item()
            runningLoss_generator += generatorLoss.item()
        
        # Work on Samples
        # import pdb; pdb.set_trace()
        baseDir = '/usr/sci/cibc/ProjectsAndScratch/DeekshithMLECG/explainability'
        samples_dir = f'{baseDir}/testAbnormalData/hundred_samples.pt'
        samples = torch.load(samples_dir, weights_only=True, map_location=torch.device("cpu"))
        
        modifications = generator(samples['ECGs'])
        
        samples['modifications'] = modifications.cpu()
        samples['generated_ecg'] = modificationFunction(samples['ECGs'], samples['modifications'])
        
        matlab_data = {
            "ECGs": samples["ECGs"].numpy().astype(float),
            "KCLs": samples["KCLs"].numpy().astype(float),
            "Paths": np.array(samples["PATHS"], dtype=object),  # Paths as an object array
            "Modifications": samples["modifications"].numpy().astype(float),
            "Generated_ECGs": samples['generated_ecg'].numpy().astype(float)
        }
        
        ECGs = matlab_data['ECGs']
        generated_ecgs = matlab_data['Generated_ECGs']
        for i in range(ECGs.shape[0]):
            ecg = ECGs[i].squeeze(0)
            generated = generated_ecgs[i].squeeze(0)
            matlab_data[f'ecg_{i}'] = ecg
            matlab_data[f'generated_ecg_{i}'] = generated
        
        directory = f'{directory}/samples/'
        os.makedirs(directory, exist_ok=True)
        os.makedirs(f'{directory}/modifications/', exist_ok=True)
        torch.save(samples, f'{directory}/samples_{epoch}.pt')
        # torch.save(matlab_data, f'{directory}/matlab_{epoch}.pt')
        savemat(f'{directory}/matlab_{epoch}.mat', matlab_data)
        
        with open(f'{directory}/modifications/modifications_{epoch}.json', 'w') as json_file:
            json.dump(modifications.tolist(), json_file)
        
        # Work on 4 Example Data
        indices = [76, 40, 99, 36]

        ECGs = torch.stack([samples['ECGs'][i] for i in indices])
        ECGs = ECGs.to(device)
        KCLs = torch.stack([samples['KCLs'][i] for i in indices])
        modifications = generator(ECGs)
        
        if figNeeded:
            print("Creating Figure")
            plt.suptitle(f'Four Example ECGs')
            for i in range(4):
                for lead in range(8):
                    axes[lead, i].title.set_text(f'D {lead}, {KCLs[i].item()}')
                    axes[lead, i].plot(ECGs[i, 0, lead,:].detach().clone().squeeze().cpu().numpy(), 'k', linewidth=1, linestyle='--')
                    axes[lead, i].plot(modifications[i, 0, lead, :].detach().clone().squeeze().cpu().numpy(), 'r', linewidth=1, linestyle='--')
                    axes[lead, i].plot(modificationFunction(ECGs[i,0,lead,:].detach().clone(),modifications[i,0,lead,:].detach().clone()).squeeze().cpu().numpy(),'g', linewidth=2)
                    
        discriminator.zero_grad()
        generator.zero_grad()
        evalLoss_wgan = runningLoss_wgan/len(testNormalData.dataset)
        evalLoss_generator = runningLoss_generator/len(testNormalData.dataset)
        plt.tight_layout()
    return evalLoss_wgan,evalLoss_generator, plt

    

def trainExplainabilityNetworks(discriminator, generator, optim_discriminator, optim_generator, modificationFunction, modificationPenalty, 
                                trainNormalData, trainAbnormalData, testNormalData, testAbnormalData, epochs, lossParams, modelSaveDir, label,
                                genIter_perEp, discrimIter_perEp, evalUpdates, lossFun, logtowandb):
    prevTrainingLoss_wgan = 0.0
    prevTrainingLoss_generator = 0.0
    prevTrainLoss_class = 0.0
    bestEval = 1e10
    datetimeStr = str(datetime.datetime.now()).replace(' ','_').replace(':','-').replace('.','-')
    dir = f"{modelSaveDir}/RUN_{datetimeStr}/"
    os.makedirs(dir, exist_ok=True)
    print('Explainibility training started')
    # problemType = discriminator.problemType
    print(f"Training Sizes: normal dataset = {len(trainNormalData.dataset)} and abnormal dataset = {len(trainAbnormalData.dataset)}")
    for ep in range(epochs):
        print(f'Begin epoch {ep}')
        discriminator.train()
        generator.train()
    
        runningLoss_wgan = 0.0
        runningLoss_generator = 0.0
        running_fake = 0.0
        running_true = 0.0
        running_genPen = 0.0
        
        normal_data_iter = iter(trainNormalData)
        abnormal_data_iter = iter(trainAbnormalData)
        
        for ix in range(len(trainNormalData)):
            print(f'Batch {ix} of {len(trainNormalData)} of epoch {ep}           ')
            
            
            # Train the discriminator
            for descrimIx in range(discrimIter_perEp[ep]):
                try:
                     normal_data, flag = next(normal_data_iter)
                except StopIteration:
                    normal_data_iter = iter(trainNormalData)
                    normal_data, flag = next(normal_data_iter)
        
                try:
                    abnormal_data, flag = next(abnormal_data_iter)
                except StopIteration:
                    abnormal_data_iter = iter(trainAbnormalData)
                    abnormal_data, flag = next(abnormal_data_iter)
                
                normal_data = normal_data.to(device)
                abnormal_data = abnormal_data.to(device)
                
                print(f'Discriminator Step {descrimIx+1} of {discrimIter_perEp[ep]}', end='\r')
                
                optim_discriminator.zero_grad()
                
                TrueResults = discriminator(normal_data)
                
                modifications = generator(abnormal_data).detach()
                fakeData = modificationFunction(abnormal_data, modifications)
                
                FakeResults = discriminator(fakeData)
                
                wganLoss = lossFun(discriminator, TrueResults, FakeResults, normal_data, abnormal_data, lossParams['weights'])
                
                runningLoss_wgan += wganLoss.item()
                wganLoss.backward()
                optim_discriminator.step()
            runningLoss_wgan = runningLoss_wgan / discrimIter_perEp[ep]
            running_fake = running_fake + FakeResults.detach().sum().item()
            running_true = running_true + TrueResults.detach().sum().item()
            print(f'Discriminator Step {descrimIx+1} of {discrimIter_perEp[ep]}')
            
            # Generator iterations
            for genIx in range(genIter_perEp[ep]):
                print(f'Generator step {genIx+1} of {genIter_perEp[ep]}', end='\r')
                
                optim_generator.zero_grad()
                
                try:
                    abnormal_data, flag = next(abnormal_data_iter)
                except StopIteration:
                    abnormal_data_iter = iter(trainAbnormalData)
                    abnormal_data, flag = next(abnormal_data_iter)
                    
                abnormal_data = abnormal_data.to(device)
                modifications = generator(abnormal_data)
                
                
                
                fakeData = modificationFunction(abnormal_data, modifications)
                FakeResults = discriminator(fakeData)
                
                modPen = modificationPenalty(modifications)
                genPen = -FakeResults.mean()
                generatorLoss = modPen*lossParams['weights']['modificationWeight'] + genPen*lossParams['weights']['generatorWeight']
                
                generatorLoss.backward()
                optim_generator.step()
                runningLoss_generator += generatorLoss.item()
            
            running_genPen = running_genPen + genPen.detach().sum().item()
            print(f'Generator step {genIx+1} of {genIter_perEp[ep]}')
            runningLoss_generator = runningLoss_generator / genIter_perEp[ep]
        
        averageTrainLoss_wgan = runningLoss_wgan / len(trainNormalData.dataset)
        averageTrainLoss_generator = runningLoss_generator / len(trainNormalData.dataset)
        
        averageFake = running_fake / len(trainNormalData.dataset)
        averageTrue = running_true / len(trainNormalData.dataset)
        averageGenPen = running_genPen / len(trainNormalData.dataset)
        print(f'Ep: {ep}, average train loss: wgan: {averageTrainLoss_wgan:.07f},diff: {averageTrainLoss_wgan-prevTrainingLoss_wgan:.07f}')
        print(f'Ep: {ep}, average train loss: gen : {averageTrainLoss_generator:.07f},diff: {averageTrainLoss_generator-prevTrainingLoss_generator:.07f}')
        
        prevTrainingLoss_wgan = averageTrainLoss_wgan
        prevTrainingLoss_generator = averageTrainLoss_generator
        trainingLog = dict(Epoch=ep,trainLoss_wgan=averageTrainLoss_wgan,trainLoss_generator=averageTrainLoss_generator,averageFake=averageFake,averageTrue=averageTrue,average_generatorPenalty=averageGenPen, trueMinusFake=averageTrue-averageFake)
        saveFlag = 0
        
        save_config = dict(
            epoch=ep,
            generator=generator.state_dict(),
            datetime=datetimeStr,
            lossParams=lossParams,
        )
        os.makedirs(f"{dir}/generator/", exist_ok=True)
        torch.save(save_config, f'{dir}/generator/generator_{ep}.pt')
        print(f'Saved model at {dir}/generator/generator_{ep}.pt')
        
        
        if ep in evalUpdates:
            print('Evaluating performance')
            evalLoss_wgan,evalLoss_generator,figure = eval_generator_discriminator(discriminator,generator,testNormalData, testAbnormalData,lossFun,lossParams,modificationFunction,modificationPenalty, directory=dir, epoch=ep)
            evalLoss = evalLoss_wgan + evalLoss_generator
            if evalLoss < bestEval:
                bestEval = evalLoss
                # human readable date and time
                
                save_config = dict(
                    epoch=ep,
                    generator=generator.state_dict(),
                    discriminator=discriminator.state_dict(),
                    optim_generator=optim_generator.state_dict(),
                    optim_discriminator=optim_discriminator.state_dict(),
                    lossParams=lossParams,
                    datetime=datetimeStr,
                )
                    
                
                # torch.save(generator.state_dict(), f'{modelSaveDir}/generator_{datetime}.pt')
                
                torch.save(save_config, f'{dir}/state_{datetimeStr}.pt')
                print(f'Saved model at {dir}/state_{datetimeStr}.pt')
                saveFlag = 1
            trainingLog['evalLoss_total'] = evalLoss
            trainingLog['evalLoss_wgan'] = evalLoss_wgan
            trainingLog['evalLoss_generator'] = evalLoss_generator
            trainingLog['example'] = figure
        
        trainingLog['saveFlag'] = saveFlag
        if logtowandb:
            print("Logging to wandb")
            wandb.log(trainingLog)
            
            
            
            