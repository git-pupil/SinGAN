from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()  # 导入基础参数配置
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images') #训练图片文件夹
    parser.add_argument('--input_name', help='input image name', required=True) #训练图片名字（带后缀）
    parser.add_argument('--mode', help='task to be done', default='train')  # 训练任务的模式
    opt = parser.parse_args()
    opt = functions.post_config(opt) #对参数做进一步处理
    Gs = []  # 生成器列表
    Zs = []  # 噪声列表
    reals = []  # 每一层的真实图像（大小各不相同），用于进行模型训练的损失计算
    NoiseAmp = []  # ？？
    dir2save = functions.generate_dir2save(opt)  # 获取存储图像的路径

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        # real shape: [batch, channel, w, h]
        #          -> [1, 3, w, h]
        real = functions.read_image(opt)  # 以tensor格式读取输入图像
        functions.adjust_scales2image(real, opt)  # 按照输入图片的尺寸获取网络层数(scales)
        train(opt, Gs, Zs, reals, NoiseAmp)  # 开始训练
        SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)  # 用训练好的模型进行图像生成
