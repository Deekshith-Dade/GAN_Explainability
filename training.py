import torch
import matplotlib.pyplot as plt
import wandb
from itertools import cycle
import sys
import datetime
import os

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

def eval_generator_discriminator(discriminator, generator, testNormalData, testAbnormalData, lossFun, lossParams, modificationFunction, modificationPenalty):
    discriminator.eval()
    generator.eval()
    plt.figure(1)  
    subDims = [8, 2]
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
            if figNeeded and not diseaseFound:
                print(' ')
                print('Looking for example data')
                plt.suptitle('Example Data')
                diseaseIx = torch.argmax(kclVals)
                diseaseVal = kclVals[diseaseIx]
                
                randIx = torch.randint(0, abnormalData.shape[0], (1,))[0]
                randVal = kclVals[randIx]
                # Mention the diseaseVal in the plots title
                fig.suptitle(f'Disease Value: {diseaseVal.item()}')
                if diseaseIx.nelement()!=0:
                    if not diseaseFound:
                        print('Found Example Disease Data')
                        diseaseFound = 1
                        for lead in range(8):
                            axes[lead, 0].title.set_text(f'D {lead}, {diseaseVal}')
                            axes[lead, 0].plot(abnormalData[diseaseIx, 0, lead,:].detach().clone().squeeze().cpu().numpy(), 'k', linewidth=1, linestyle='--')
                            axes[lead, 0].plot(modifications[diseaseIx, 0, lead, :].detach().clone().squeeze().cpu().numpy(), 'r', linewidth=1, linestyle='--')
                            axes[lead, 0].plot(modificationFunction(abnormalData[diseaseIx,0,lead,:].detach().clone(),modifications[diseaseIx,0,lead,:].detach().clone()).squeeze().cpu().numpy(),'g', linewidth=2)
                            
                            axes[lead, 1].title.set_text(f'D {lead}, {randVal}')
                            axes[lead, 1].plot(abnormalData[randIx, 0, lead,:].detach().clone().squeeze().cpu().numpy(), 'k', linewidth=1, linestyle='--')
                            axes[lead, 1].plot(modifications[randIx, 0, lead, :].detach().clone().squeeze().cpu().numpy(), 'r', linewidth=1, linestyle='--')
                            axes[lead, 1].plot(modificationFunction(abnormalData[randIx,0,lead,:].detach().clone(),modifications[randIx,0,lead,:].detach().clone()).squeeze().cpu().numpy(),'g', linewidth=2)
                            

                            
                
                # normalIxs = torch.randint(0, normalData.shape[0], (1,))
                # if normalIxs.nelement() != 0:
                #     if not healthFound:
                #         print('Found example healthy')
                #         healthFound = 1
                #         for lead in range(8):
                #             axes[lead,1].plot(normalData[normalIxs[0],0,lead,:].detach().clone().squeeze().cpu().numpy(),'k')
                            
                
                if healthFound == 1 and diseaseFound == 1:
                    figNeeded = 0
                print(' ')
    
        
        print(' ')			
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
    os.makedirs(modelSaveDir, exist_ok=True)
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
        if ep in evalUpdates:
            print('Evaluating performance')
            evalLoss_wgan,evalLoss_generator,figure = eval_generator_discriminator(discriminator,generator,testNormalData, testAbnormalData,lossFun,lossParams,modificationFunction,modificationPenalty)
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
                torch.save(save_config, f'{modelSaveDir}/state_{datetimeStr}.pt')
                print(f'Saved model at {modelSaveDir}/state_{datetimeStr}.pt')
                saveFlag = 1
            trainingLog['evalLoss_total'] = evalLoss
            trainingLog['evalLoss_wgan'] = evalLoss_wgan
            trainingLog['evalLoss_generator'] = evalLoss_generator
            trainingLog['example'] = figure
        
        trainingLog['saveFlag'] = saveFlag
        if logtowandb:
            print("Logging to wandb")
            wandb.log(trainingLog)
            
            
            
            