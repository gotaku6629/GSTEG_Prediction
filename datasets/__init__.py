""" Initilize the datasets module
    New datasets can be added with python scripts under datasets/

    18枚の入力画像に対して, GAP=4で10枚セットを3層作る.
"""
import torch
import torch.utils.data
import torch.utils.data.distributed
import importlib


def get_dataset(args):

    dataset = importlib.import_module('.'+args.dataset, package='datasets')
    #print('dataset=', dataset) # datasets/charades.py

    print('2.1 dataset.get') #224x224
    train_dataset, val_dataset, valvideo_dataset = dataset.get(args)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None


    print('2.2 DataLoader') # 25x224x224
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    valvideo_loader = torch.utils.data.DataLoader(
        valvideo_dataset, batch_size=valvideo_dataset.testGAP, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, valvideo_loader
