import numpy as np
from sklearn.metrics import r2_score, explained_variance_score

import torch
import torchvision.transforms as transforms

from baseline_models import *

def model_loader(model_arch):

    model_dict={'scnn':shallow_cnn(), 'r18':Resnet18(), 'r34':Resnet34(), 'r50':Resnet50(), 'r101':Resnet101(), 'r152':Resnet152()}
    model = model_dict[model_arch]

    return model

# def transform_batch_input(x, mode):

#     if mode=='train':
#         normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#     transformList = []
    
#     transformList.append(transforms.ToPILImage())
#     if mode=='train':
#         transformList.append(transforms.RandomCrop(224))
#     else:
#         transformList.append(transforms.Resize(224))
#     transformList.append(transforms.ColorJitter(brightness=0, contrast=0.25))
#     transformList.append(transforms.ToTensor())
#     if mode=='train':
#         transformList.append(normalize)      
    
#     transformSequence=transforms.Compose(transformList)
    
#     y = torch.zeros(size=(x.shape[0], x.shape[1], 224, 224), dtype = torch.float32)
    
#     for i in range(x.shape[0]):
#         y[i] = transformSequence(x[i])
    
#     return y

def compute_metrics(y_true, y_pred):

    scores=[]

    combined_r2 = r2_score(y_true, y_pred, multioutput='variance_weighted')
    combined_evs = explained_variance_score(y_true, y_pred, multioutput='variance_weighted')

    return combined_r2, combined_evs

def train_epoch(model, train_generator, criterion, optimizer, model_type, device):

    model.train()

    runningLoss = 0
    
    for i, (inputs, targets) in enumerate(train_generator):

        if i:
            y_true=np.vstack([y_true, targets.cpu().numpy()])
        else:
            y_true=targets.cpu().numpy()
        
        # Initialize gradients to zero
        optimizer.zero_grad()  
        
        with torch.set_grad_enabled(True):
            # Feed-forward
            if model_type!='rgbd_cnn':
                output = model(inputs.float().to(device))
            else:
                output = model(inputs['rgb'].float().to(device), inputs['depth'].float().to(device))
            # Add batch outputs
            if i:
                y_pred = np.vstack([y_pred, output.detach().cpu().numpy()])
            else:
                y_pred=output.detach().cpu().numpy()

            targets=targets.float().to(device)
            # Calculate Loss
            loss = criterion(output, targets)
            # accumulate loss
            runningLoss += loss.item()
            # Backpropagate loss and compute gradients
            loss.backward()
            # Update the network parameters
            optimizer.step()

    train_r2, train_evs = compute_metrics(y_true, y_pred)
        
    return model, runningLoss/(len(train_generator)), (train_r2, train_evs)

def val_epoch(model, val_generator, criterion, model_type, device):

    model.eval()

    runningLoss = 0
    
    for i, (inputs, targets) in enumerate(val_generator):

        if i:
            y_true=np.vstack([y_true, targets.cpu().numpy()])
        else:
            y_true=targets.cpu().numpy()
        
        with torch.set_grad_enabled(False):
            # Feed-forward
            if model_type!='rgbd_cnn':
                output = model(inputs.float().to(device))
            else:
                output = model(inputs['rgb'].float().to(device), inputs['depth'].float().to(device))
            # Add batch outputs
            if i:
                y_pred = np.vstack([y_pred, output.cpu().numpy()])
            else:
                y_pred=output.cpu().numpy()

            targets=targets.float().to(device)
            # Calculate Loss
            loss = criterion(output, targets)
            # accumulate loss
            runningLoss += loss.item()

    val_r2, val_evs = compute_metrics(y_true, y_pred)
        
    return runningLoss/(len(val_generator)), (val_r2, val_evs)

def test_model(model, test_generator, model_type, device):

    model.eval()
    
    for i, (inputs, targets) in enumerate(test_generator):

        if i:
            y_true=np.vstack([y_true, targets.cpu().numpy()])
        else:
            y_true=targets.cpu().numpy()
        
        with torch.no_grad():
            # Feed-forward
            if model_type!='rgbd_cnn':
                output = model(inputs.float().to(device))
            else:
                output = model(inputs['rgb'].float().to(device), inputs['depth'].float().to(device))
            # Add batch outputs
            if i:
                y_pred = np.vstack([y_pred, output.cpu().numpy()])
            else:
                y_pred=output.cpu().numpy()

    test_r2, test_evs = compute_metrics(y_true, y_pred)
        
    return test_r2, test_evs