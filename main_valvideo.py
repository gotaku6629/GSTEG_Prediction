#!/usr/bin/env python

"""Charades activity recognition baseline code
   Using Validation Video in this model

   ** change the index in charades.py def_getitem__
   from s_target = self.data['s_targets'][index+1]
   to   s_target = self.data['s_targets'][index]
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


def seed(manualseed):
    random.seed(manualseed)
    np.random.seed(manualseed)
    torch.manual_seed(manualseed)
    torch.cuda.manual_seed(manualseed)


best_mAP = 0
def main():
    global opt, best_mAP
    opt = parse()
    tee.Tee(opt.cache+'/log_0724-valvideo.txt')
    #print(vars(opt))
    seed(opt.manual_seed)

    print('1. create_model')
    base_model, logits_model, criterion, base_optimizer, logits_optimizer = create_model(opt)
    if opt.resume:
        print('checkpoints load')
        #best_mAP = checkpoints.load(opt, base_model, logits_model, base_optimizer, logits_optimizer)
        checkpoints.load(opt, base_model, logits_model, base_optimizer, logits_optimizer)
    
    #print('base_model = InceptionI3D Networks') # InceptionI3D Networks
    #print(base_model)
    #print('logits_model = AsyncTFBase: Linear Networks') # AsyncTFBase: Linear Networks
    #print(logits_model)
    
    trainer = train.Trainer()

    print('2. get_dataset')
    train_loader, val_loader, valvideo_loader = get_dataset(opt)
    #print('train_loader') # [56586, [25,img,s,v,o,meta]]
    #print(train_loader)    # 56586=pairs
    #print('val_loader')   # [12676, [25,img,s,v,o,meta]]
    #print(val_loader)
    #print('valvideo_loader') # [1863, [25+1,img,s,v,o,meta]]
    #print(valvideo_loader)   # 1863=num_(kind of video)

    if opt.evaluate:
        trainer.validate(val_loader, base_model, logits_model, criterion, -1, opt)
        trainer.validate_video(valvideo_loader, base_model, logits_model, criterion, -1, opt)
        return

    print('3.3 Valiation Video')
    #if opt.distributed:
    #    trainer.train_sampler.set_epoch(epoch)
        
    sov_mAP, sov_rec_at_n, sov_mprec_at_n  = trainer.validate_video(valvideo_loader, base_model, logits_model, criterion, epoch, opt)
        
    is_best = sov_mAP > best_mAP
    best_mAP = max(sov_mAP, best_mAP)
    scores = {'mAP':sov_mAP,'sov_rec_at_n':sov_rec_at_n,'sov_mprec_at_n':sov_mprec_at_n}
    checkpoints.score_file(scores, "{}/model_{}.txt".format(opt.cache, 'valvideo'))


if __name__ == '__main__':
    main()
