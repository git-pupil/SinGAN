from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions

#图像融合应用
if __name__ == '__main__':
    # 输入参数
    parser = get_arguments()
    # 训练图像
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='training image name', required=True)
    # 输入图像（需要协调的图像）
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Harmonization')
    parser.add_argument('--ref_name', help='reference image name', required=True)
    # 开始融合的层数
    parser.add_argument('--harmonization_start_scale', help='harmonization injection scale', type=int, required=True)

    parser.add_argument('--mode', help='task to be done', default='harmonization')
    opt = parser.parse_args()
    opt = functions.post_config(opt) # 初始化固定的参数
    # 定义变量
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt) # 生成图像保存的路径
    if dir2save is None:
        print('task does not exist')
    #elif (os.path.exists(dir2save)):
    #    print("output already exist")
    else:
        try:
            os.makedirs(dir2save) # 创建文件夹
        except OSError:
            pass
        # 加载训练图片
        real = functions.read_image(opt)
        real = functions.adjust_scales2image(real, opt) # 缩放图像至指定大小
        # 加载训练好的模型
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        if (opt.harmonization_start_scale < 1) | (opt.harmonization_start_scale > (len(Gs)-1)):
            # 要求输入的参数在训练尺寸的范围内
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else:
            # 加载需要协调的图片
            ref = functions.read_image_dir('%s/%s' % (opt.ref_dir, opt.ref_name), opt)
            # 加载掩模图片
            mask = functions.read_image_dir('%s/%s_mask%s' % (opt.ref_dir,opt.ref_name[:-4],opt.ref_name[-4:]), opt)
            # 调整输入/掩模图像和训练图像的尺寸相同（最高层尺寸）
            if ref.shape[3] != real.shape[3]:
                mask = imresize_to_shape(mask, [real.shape[2], real.shape[3]], opt)
                mask = mask[:, :, :real.shape[2], :real.shape[3]]
                ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]
            # 对掩模做膨胀处理
            mask = functions.dilate_mask(mask, opt)

            N = len(reals) - 1 # 金字塔层数序标
            n = opt.harmonization_start_scale # 开始加入融合图片的层数
            # 调整输入图像尺寸与injection scale层一致
            in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            # 生成图像
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            out = (1-mask)*real+mask*out # 背景部分保留原图像，掩模部分使用生成的图像
            # 保存图像
            plt.imsave('%s/start_scale=%d.png' % (dir2save,opt.harmonization_start_scale), functions.convert_image_np(out.detach()), vmin=0, vmax=1)




