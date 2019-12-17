import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import argparse
import time
import copy
import os
import numpy as np
import pandas as pd

from utils import *
from sequence_models import *
from dataloader import LSTM_seq_loader 

import torch.backends.cudnn as cudnn

def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_device)
    cudnn.benchmark = True

    # Seed the random states
    np.random.seed(0)
    random_state  = np.random.RandomState(0)

    # Random seed for torch
    torch.manual_seed(args.rand_seed)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nDevice:{}\n'.format(device))

    if args.operation==0:
        base_path='/storage/soumava/JIGSAWS/Knot_Tying/npy_files/'
        save_path='/storage/soumava/JIGSAWS/Knot_Tying/saved_results/'+args.model_arch+'/'
    elif args.operation==1:
        base_path='/storage/soumava/JIGSAWS/Needle_Passing/npy_files/'
        save_path='/storage/soumava/JIGSAWS/Needle_Passing/saved_results/'+args.model_arch+'/'
    elif args.operation==2:
        base_path='/storage/soumava/JIGSAWS/Suturing/npy_files/'
        save_path='/storage/soumava/JIGSAWS/Suturing/saved_results/'+args.model_arch+'/'

    if args.capture==1:
        rgb_path='capture1/frames/'
        labels_path='capture1/labels/'
        
    elif args.capture==2:
        rgb_path='capture2/frames/'
        labels_path='capture2/labels/'

    else:
        train_rgb=None
        val_rgb=None
        test_rgb=None
        labels_path='capture1/labels/'

    if args.capture==1 or args.capture==2:
        train_rgb=base_path+'train/'+rgb_path
        val_rgb=base_path+'val/'+rgb_path
        test_rgb=base_path+'test/'+rgb_path

    if args.model_type!='rgb_cnn':
        train_d=base_path+'train/depth_maps/'
        val_d=base_path+'val/depth_maps/'
        test_d=base_path+'test/depth_maps/'
    else:
        train_d=None
        val_d=None
        test_d=None

    # Image transformations
    image_transforms = {
        'train':
        transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0, contrast=0.25),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # Imagenet standards
        ]),
        # Validation does not use augmentation
        'eval':
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_params = {'batch_size':args.batch_size, 'shuffle':True, 'num_workers':24}
    eval_params = {'batch_size':args.batch_size, 'shuffle':False, 'num_workers':6}

    data={
    'train':LSTM_seq_loader(rgb_path=train_rgb, labels_path=base_path+'train/'+labels_path, depth_path=train_d, model_type=args.model_type, resize_dim=256,
                            context_len=args.context_len, transform=image_transforms['train']),
    'val':LSTM_seq_loader(rgb_path=val_rgb, labels_path=base_path+'val/'+labels_path, depth_path=val_d, model_type=args.model_type, resize_dim=224, 
                          context_len=args.context_len, transform=image_transforms['eval']),
    'test':LSTM_seq_loader(rgb_path=test_rgb, labels_path=base_path+'test/'+labels_path, depth_path=test_d, model_type=args.model_type, resize_dim=224, 
                           context_len=args.context_len, transform=image_transforms['eval'])
    }

    print('#Train samples:{}'.format(len(data['train'])))
    print('#Val samples:{}'.format(len(data['val'])))
    print('#Test samples:{}\n'.format(len(data['test'])))

    dataloaders={
    'train':DataLoader(data['train'], **train_params),
    'val':DataLoader(data['val'], **eval_params),
    'test':DataLoader(data['test'], **eval_params)
    }
    
    if args.model_type!='rgbd_cnn':
        if args.capture==1 or 2:
            wts=torch.load(save_path+'capture'+str(args.capture)+\
                '/val_r20.3461_test_r20.4099_test_evs0.4259_train_r20.5907_model-rgb_cnn_arch-r18-bEp-8_cap-1_e-50_bs-64_lr-0.001.pth')
        else:
            wts=torch.load(save_path+'depth_only/'+\
                'val_r2-0.1869_test_r20.2781_test_evs0.3131_train_r20.6085_model-depth_cnn_arch-r18_bEp-19_cap-0_e-50_bs-64_lr-0.001.pth')
        cnn=model_loader(args.model_arch)
        cnn.load_state_dict(wts)
        model=LSTM_Decoder(cnn_backbone=cnn, model_type=args.model_type, hidden_dim_ratio=args.hidden_dim_ratio, num_layers=args.num_layers,  
                           bidir=args.bidir, drop_rate=args.drop_rate, classCount=38)
    else:
        cnn_backbone=model_loader(args.model_arch)
        cnn=RGBD_CNN(cnn_backbone=cnn_backbone, classCount=38)
        wts=torch.load(save_path+'capture'+str(args.capture)+'+depth/'+'')
        cnn.load_state_dict(wts)
        model=LSTM_Decoder(cnn_backbone=cnn, model_type=args.model_type, hidden_units=args.hidden_dim_ratio, num_layers=args.num_layers,  
                           bidir=args.bidir, drop_rate=args.drop_rate, classCount=38)
    
    model.to(device)
    print('Model loading done\n')
    
    if args.backbone_trainable:
        params=list(cnn.parameters())+list(model.parameters())
    else:
        params=model.parameters()
    
    optimizer = optim.Adam(params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.l2)
    criterion = nn.MSELoss()
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, verbose=True, threshold=1e-3, min_lr=1e-5)

    best_val = -100
    training_loss=[]
    valid_loss=[]

    train_r2=[]
    train_evs=[]
    val_r2=[]
    val_evs=[]

    for epoch in range(args.num_epochs):

        since = time.time()

        # shuffled_indices = torch.randperm(train_frames.shape[0])
        # shuffled_frames = train_frames[shuffled_indices]
        # shuffled_labels = torch.from_numpy(train_labels)[shuffled_indices]
        
        model, train_loss, train_score = train_epoch(model=model, train_generator=dataloaders['train'], criterion=criterion, optimizer=optimizer, 
                                                     model_type=args.model_type, device=device)
                                                      
        model_wts_epoch = copy.deepcopy(model.state_dict())
        
        val_loss, val_score = val_epoch(model=model, val_generator=dataloaders['val'], criterion=criterion, model_type=args.model_type, device=device)
        
        training_loss.append(train_loss)
        valid_loss.append(val_loss)

        lr_scheduler.step(val_loss)

        train_r2.append(train_score[0])
        train_evs.append(train_score[1])
        val_r2.append(val_score[0])
        val_evs.append(val_score[1])
        
        if val_score[0]>best_val:
            best_val = val_score[0]
            corr_evs = val_score[1]
            bestEp = epoch+1
            best_model_wts = model_wts_epoch

        time_elapsed = time.time() - since

        print('Epoch:{}/{}, Time Elapsed:{:.0f}m {:.0f}s'.format(epoch+1, args.num_epochs, time_elapsed // 60, time_elapsed % 60))
        print('Training loss:{}, Validation loss:{}; '.format(train_loss, val_loss))
        print('Train R2:{}; Validation R2:{}'.format(train_score[0], val_score[0]))
        print('Train EVS:{}; Validation EVS:{}\n'.format(train_score[1], val_score[1]))

    print('Best performance at epoch {} with Validation R2:{} and EVS:{}.\n'.format(bestEp, best_val, corr_evs))

    print('Testing with best model\n')
    model.load_state_dict(best_model_wts)
        
    test_r2, test_evs = test_model(model=model, test_generator=dataloaders['test'], model_type=args.model_type, device=device)
    print('Test R2:{} and EVS:{}'.format(test_r2, test_evs))

    df_save = pd.DataFrame({'train_loss':training_loss, 'val_loss':valid_loss, 'train_r2':train_r2, 'val_r2':val_r2, 'train_evs':train_evs,
                            'val_evs':val_evs})

    best_train_score = train_r2[val_r2.index(best_val)]

    score_string = 'val_r2(' + str(float("{0:.4f}".format(best_val))) + ')_test_r2(' + str(float("{0:.4f}".format(test_r2)))+\
                    ')_test_evs(' + str(float("{0:.4f}".format(test_evs))) + ')_train_r2' + str(float("{0:.4f}".format(best_train_score)))
    
    suffix = score_string + \
        '_bEp-' + str(bestEp) + \
        '_e-' + str(args.num_epochs) + \
        '_bs-' + str(args.batch_size) + \
        '_lr-' + str(args.learning_rate) +\
        '_l2-' + str(args.l2)+\
        '_bt-' + str(args.backbone_trainable) +\
        '_hid-' + str(args.hidden_dim_ratio)+\
        '_lay-' + str(args.num_layers) +\
        '_bi-' + str(args.bidir) +\
        '_drop-' + str(args.drop_rate) +\
        '_con-' +str(args.context_len)

    if args.model_type!='rgbd_cnn':
        if args.capture==1 or 2:
            save_path+='lstm+cnn+cap'+str(args.capture)+'/'
        else:
            save_path+='lstm+cnn+depth_only/'
    else:
        save_path+='lstm+cnn+cap'+str(args.capture)+'+depth/'

    if args.type_of_run==1:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df_save.to_csv(open(save_path+suffix+'.csv', 'w'))
        torch.save(best_model_wts, save_path+suffix+'.pth')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="CNN Regressor Baselines for predicting kinematic variables.")
    parser.add_argument('-run', '--type_of_run', help="0 is for a dummy run, no results are saved to disk", default=1, type=int)
    parser.add_argument('-type', '--model_type', help="select b/w depth/rgb/rgbd cnn", default='rgbd_cnn', type=str)
    parser.add_argument('-arch', '--model_arch', help="select b/w diff resnet versions", default='r18', type=str)
    parser.add_argument('-bt', '--backbone_trainable', help='flag to keep cnn backbone trainable', default=True, type=bool)
    parser.add_argument('-op', '--operation', help="choose between 0(knot-tying), 1(needle-passing), 2(suturing)", default=0, type=int)
    parser.add_argument('-cap', '--capture', help="choose between 1(left cam) and 2(right cam) or 0(depth map only)", default=1, type=int)
    parser.add_argument('-e', '--num_epochs', help="number of epochs to run", default=50, type=int)
    parser.add_argument('-bs', '--batch_size', help="batch size used throughout", default=64, type=int)
    parser.add_argument('-lr', '--learning_rate', help="learning rate for adam/sgd", default=0.001, type=float)
    parser.add_argument('-l2', '--l2', help="l2 regularization for adam optimizer", default=1e-4, type=float)
    parser.add_argument('-con', '--context_len', help="#frames to take before current frame as context", default=10, type=int)
    parser.add_argument('-hid', '--hidden_dim_ratio', help="dimension of lstm hidden states", default=2, type=int)
    parser.add_argument('-lay', '--num_layers', help='number of layers in lstm', default=1, type=int)
    parser.add_argument('-bi', '--bidir', help='bidirectional flag for lstm', default=False, type=bool)
    parser.add_argument('-drop', '--drop_rate', help='dropout probability for lstm layers (except last layer)', default=0.0, type=float)
    parser.add_argument('-seed', '--rand_seed', help='seed to set manual_seed in torch', default=1, type=int)
    parser.add_argument('-cd', '--cuda_device', help='choose gpu no.', default=0, type=int)
    
    args = parser.parse_args()
    print(args)
    main(args)
