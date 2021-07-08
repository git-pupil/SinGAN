import argparse

def get_arguments():
    """给程序配置基础的命令行参数，用以控制程序运行时的不同参数。"""
    parser = argparse.ArgumentParser()
    # 工作环境
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)  # 是否不使用cuda

    # 加载、输入和保存配置
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")  # 生成器的路径
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")  # 判别器的路径
    parser.add_argument('--manualSeed', type=int, help='manual seed')  # 手动设置的随机种子
    parser.add_argument('--nc_z', type=int, help='noise # channels', default=3)  # 噪声图片的通道数
    parser.add_argument('--nc_im', type=int, help='image # channels', default=3)  # 训练图像的通道数
    parser.add_argument('--out', help='output folder', default='Output')  # 输出图像文件夹路径

    # 网络结构超参数
    parser.add_argument('--nfc', type=int, default=32)  # 全卷积层中的输入通道数
    parser.add_argument('--min_nfc', type=int, default=32)  # 全卷积中的最小输入通道数
    parser.add_argument('--ker_size', type=int, help='kernel size', default=3)  # 卷积核的尺寸
    parser.add_argument('--num_layer', type=int, help='number of layers', default=5)  # 每个模块中全卷积层的层数
    parser.add_argument('--stride', help='stride', default=1)  # 步长
    parser.add_argument('--padd_size', type=int, help='net pad size', default=0)  # math.floor(opt.ker_size/2) 填充大小

    # 金子塔参数
    parser.add_argument('--scale_factor', type=float, help='pyramid scale factor', default=0.75)  # pow(0.5,1/6)) 用于计算每层图像的尺度
    parser.add_argument('--noise_amp', type=float, help='addative noise cont weight', default=0.1)  # 用于计算噪声的权重
    parser.add_argument('--min_size', type=int, help='image minimal size at the coarser scale', default=25)  # 生成图像的最小尺寸
    parser.add_argument('--max_size', type=int, help='image minimal size at the coarser scale', default=250)  # 生成图像的最大尺寸

    # 优化器超参数
    parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train per scale')  # epoch
    parser.add_argument('--gamma', type=float, help='scheduler gamma', default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--Gsteps', type=int, help='Generator inner steps', default=3)  # 生成器训练的迭代步数
    parser.add_argument('--Dsteps', type=int, help='Discriminator inner steps', default=3)  # 判别器训练的迭代步数
    parser.add_argument('--lambda_grad', type=float, help='gradient penelty weight', default=0.1)
    parser.add_argument('--alpha', type=float, help='reconstruction loss weight', default=10)

    return parser
