#!/usr/bin/env python

"""Charades activity recognition baseline code
   Can be run directly or throught config scripts under exp/

   Gunnar Sigurdsson, 2018
""" 
import torch
import numpy as np
import random
import train
from models import create_model
from datasets import get_dataset
import checkpoints
from opts import parse
from utils import tee
import csv


def seed(manualseed):
    random.seed(manualseed)
    np.random.seed(manualseed)
    torch.manual_seed(manualseed)
    torch.cuda.manual_seed(manualseed)


best_mAP = 0
def main():
    global opt, best_mAP
    opt = parse()
    tee.Tee(opt.cache+'/log0819-t2f51.txt')
    #print(vars(opt))
    seed(opt.manual_seed)

    print('1. create_model')
    base_model, logits_model, criterion, base_optimizer, logits_optimizer = create_model(opt)
    if opt.resume:
        print('checkpoints load')
        best_mAP = checkpoints.load(opt, base_model, logits_model, base_optimizer, logits_optimizer)
        #checkpoints.load(opt, base_model, logits_model, base_optimizer, logits_optimizer)
    
    print('base_model = InceptionI3D Networks') # InceptionI3D Networks
    #print(base_model)
    print('logits_model = AsyncTFBase: Linear Networks') # AsyncTFBase: Linear Networks
    #print(logits_model)
    
    trainer = train.Trainer()

    print('2. get_dataset')
    train_loader, val_loader, valvideo_loader = get_dataset(opt)
    #print('train_loader') # [56586, [25, img, tuple]]
    #print(train_loader)    # 56586のペア(img-tuple)
    #print('val_loader')   # [12676, [25, img, tuple]]
    #print(val_loader)
    #print('valvideo_loader') # [1863, [25, img, tuple]]
    #print(valvideo_loader)   # 1863=ビデオの種類

    if opt.evaluate:                                             
        trainer.validate(val_loader, base_model, logits_model, criterion, -1, opt)
        trainer.validate_video(valvideo_loader, base_model, logits_model, criterion, -1, opt)
        return

   # write csv
    with open('train_log.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['i', 'loss', 's', 'v', 'o'])

        print('3. Train & Test (Validation)')
        for epoch in range(opt.start_epoch, opt.epochs): # 0~20
            #print('epoch = ', epoch)
            if opt.distributed:
                trainer.train_sampler.set_epoch(epoch)
            
            print('3.1 Training')
            s_top1,s_top5,o_top1,o_top5,v_top1,v_top5, sov_top1 = trainer.train(train_loader, base_model, logits_model, criterion, base_optimizer, logits_optimizer, epoch, opt, csv_writer)
            
            print('3.2 Test (Validation)')
            s_top1val,s_top5val,o_top1val,o_top5val,v_top1val,v_top5val, sov_top1val = trainer.validate(val_loader, base_model, logits_model,  criterion, epoch, opt)
            
            print('3.3 Test (Validation_Video)')
            sov_mAP, sov_rec_at_n, sov_mprec_at_n  = trainer.validate_video(valvideo_loader, base_model, logits_model, criterion, epoch, opt)
            
            is_best = sov_mAP > best_mAP
            best_mAP = max(sov_mAP, best_mAP)
            scores = {'s_top1':s_top1,'s_top5':s_top5,'o_top1':o_top1,'o_top5':o_top5,'v_top1':v_top1,'v_top5':v_top5,'sov_top1':sov_top1,'s_top1val':s_top1val,'s_top5val':s_top5val,'o_top1val':o_top1val,'o_top5val':o_top5val,'v_top1val':v_top1val,'v_top5val':v_top5val,'sov_top1val':sov_top1val,'mAP':sov_mAP,'sov_rec_at_n':sov_rec_at_n,'sov_mprec_at_n':sov_mprec_at_n}
            #scores = {'s_top1':s_top1,'s_top5':s_top5,'o_top1':o_top1,'o_top5':o_top5,'v_top1':v_top1,'v_top5':v_top5,'sov_top1':sov_top1,'s_top1val':s_top1val,'s_top5val':s_top5val,'o_top1val':o_top1val,'o_top5val':o_top5val,'v_top1val':v_top1val,'v_top5val':v_top5val,'sov_top1val':sov_top1val}
            checkpoints.save(epoch, opt, base_model, logits_model, base_optimizer, logits_optimizer, is_best, scores)
            #checkpoints.save(epoch, opt, base_model, logits_model, base_optimizer, logits_optimizer, scores)


if __name__ == '__main__':
    main()
