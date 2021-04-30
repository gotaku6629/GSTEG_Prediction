#!/usr/bin/env python
import sys
import pdb
import traceback
#sys.path.insert(0, '..')
sys.path.insert(0, '.')
from main import main
#from main_valvideo import main
from bdb import BdbQuit
import subprocess
subprocess.Popen('find ./exp/.. -iname "*.pyc" -delete'.split())

args = [
    '--future', '51', # time of predicting future flame
    '--temporal',  '3', # time of using ST-Graph
    '--name', 'test_f51',
    '--cache-dir', './cr_caches/',
    '--rgb-data', './gsteg/Charades_v1_rgb/', # videos
    '--rgb-pretrained-weights', './gsteg/rgb_i3d_pretrained.pt', # I3D pretrained file 
    '--resume', './cr_caches/test_f51/model.pth.tar', # result write & save file
    '--train-file', './gsteg/Charades/Charades_v1_train.csv',
    '--val-file', './gsteg/Charades/Charades_v1_test.csv',
    '--groundtruth-lookup', './utils/groundtruth.p'    
#'--evaluate',
]
sys.argv.extend(args)
try:
    main()
except BdbQuit:
    sys.exit(1)
except Exception:
    traceback.print_exc()
    print('')
    pdb.post_mortem()
    sys.exit(1)


