from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions

# 真实图像的绘制转换应用在目标真实图像的预训练模型基础上进行
# 将预训练模型的第n层将生成器的输入图像更改为对应尺寸的手绘风格图进行生成
# 参数意义：
#     input_name: 需要绘制的图像，这里要求已有该图像的预训练模型
#     ref_name  : 最终目标绘制结果
#     paint_start_scale: 开始绘制的层数，在该层之前的模型不会被调用，后续也称为注入层（Injection Layer)
#     quantization_flag: 该值默认为0(False)， 其值为1(True)时需要对图像进行来量化操作

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='training image name', required=True)
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Paint')
    parser.add_argument('--ref_name', help='reference image name', required=True)
    parser.add_argument('--paint_start_scale', help='paint injection scale', type=int, required=True)
    parser.add_argument('--quantization_flag', help='specify if to perform color quantization training', type=bool, default=False)
    parser.add_argument('--mode', help='task to be done', default='paint2image')

    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []

    # 生成保存路径，路径不存在说明当前图像的预训练模型不存在
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        # 读取并生成图像金字塔
        real = functions.read_image(opt)
        real = functions.adjust_scales2image(real, opt)

        # 加载预训练模型
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)

        # 判断参考图像注入层，即开始生成的层
        if (opt.paint_start_scale < 1) | (opt.paint_start_scale > (len(Gs)-1)):
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else:
            ref = functions.read_image_dir('%s/%s' % (opt.ref_dir, opt.ref_name), opt)

            # 调整二者尺寸至相同大小（此时为最高尺度图）
            if ref.shape[3] != real.shape[3]:
                ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]

            N = len(reals) - 1
            n = opt.paint_start_scale

            # 调整参考图像尺寸使其符合注入层大小
            in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]

            # 当输出需要量化时需要重新训练模型
            # 在训练前对真实图像和参考图像进行尺寸调整（注入层大小）和色彩量化
            if opt.quantization_flag:
                opt.mode = 'paint_train'
                dir2trained_model = functions.generate_dir2save(opt)

                # 对真实图像进行尺寸调整（注入层大小）和色彩量化
                real_s = imresize(real, pow(opt.scale_factor, (N - n)), opt)
                real_s = real_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
                real_quant, centers = functions.quant(real_s, opt.device)
                plt.imsave('%s/real_quant.png' % dir2save, functions.convert_image_np(real_quant), vmin=0, vmax=1)
                plt.imsave('%s/in_paint.png' % dir2save, functions.convert_image_np(in_s), vmin=0, vmax=1)

                # 对参考图像进行色彩量化和尺寸调整，聚类中心与真实图像相同
                in_s = functions.quant2centers(ref, centers)
                in_s = imresize(in_s, pow(opt.scale_factor, (N - n)), opt)
                in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
                plt.imsave('%s/in_paint_quant.png' % dir2save, functions.convert_image_np(in_s), vmin=0, vmax=1)

                # 加载或训练已量化真实图像的SinGAN生成模型
                if (os.path.exists(dir2trained_model)):
                    Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
                    opt.mode = 'paint2image'
                else:
                    train_paint(opt, Gs, Zs, reals, NoiseAmp, centers, opt.paint_start_scale)
                    opt.mode = 'paint2image'

            # 生成并保存图像
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            plt.imsave('%s/start_scale=%d.png' % (dir2save, opt.paint_start_scale), functions.convert_image_np(out.detach()), vmin=0, vmax=1)





