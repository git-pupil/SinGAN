#============================================================
# Create Time:			2021-07-05 21:51:07
# Last modify:			2021-07-05 22:25:33
# Writer:				Wenhao	1795902848@qq.com
# File Name:			test_random_sample.py
# File Type:			PY Source File
# Tool:					Mac -- vim & python
# Information:			
#============================================================
import sys
sys.path.append('../')

from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='../Input/Images')
    parser.add_argument('--input_name', help='input image name', required=False, default='tree.png')
    parser.add_argument('--mode', help='task to be done', default='train')

    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    print(dir2save)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
    real = functions.read_image(opt)
    print(real.shape)
    functions.adjust_scales2image(real, opt)
    #train(opt, Gs, Zs, reals, NoiseAmp)
    #SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
