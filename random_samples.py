from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions

# 该方案生成与训练图像相似的随机样本，模型接收参数：
#     gen_start_scale: 模型从这一层开始生成样本并向上迭代，该层以下的层不参与生成过程
#     scale_h，scale_v: 生成图像的水平/竖直缩放指数

if __name__ == '__main__':
    """用训练好的模型生成随机图像"""
    parser = get_arguments()  # 导入基础参数配置
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='random_samples | random_samples_arbitrary_sizes', default='train', required=True)
    # for random_samples:
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    # for random_samples_arbitrary_sizes:
    parser.add_argument('--scale_h', type=float, help='horizontal resize factor for random samples', default=1.5)
    parser.add_argument('--scale_v', type=float, help='vertical resize factor for random samples', default=1)  # 导入额外参数配置
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    elif (os.path.exists(dir2save)):
        if opt.mode == 'random_samples':
            print('random samples for image %s, start scale=%d, already exist' % (opt.input_name, opt.gen_start_scale))
        elif opt.mode == 'random_samples_arbitrary_sizes':
            print('random samples for image %s at size: scale_h=%f, scale_v=%f, already exist' % (opt.input_name, opt.scale_h, opt.scale_v))
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        if opt.mode == 'random_samples':
            real = functions.read_image(opt)  # 读取训练原图
            functions.adjust_scales2image(real, opt)  # 计算网络层数
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)  # 读取训练好的模型及参数
            # 生成模型开始生成随机图像层（即gen_start_scale）的输入，如果scale=0，则为全0的数组
            in_s = functions.generate_in2coarsest(reals,1,1,opt)  
            SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale)  # 进行图像生成

        elif opt.mode == 'random_samples_arbitrary_sizes':
            real = functions.read_image(opt)  # 读取训练原图
            functions.adjust_scales2image(real, opt)  # 计算网络层数
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)  # 读取训练好的模型及参数
            in_s = functions.generate_in2coarsest(reals,opt.scale_v,opt.scale_h,opt)  # 生成上一层输出（首层为全0）
            SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, in_s, scale_v=opt.scale_v, scale_h=opt.scale_h)  # 进行图像生成





