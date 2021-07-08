# This code was taken from: https://github.com/assafshocher/resizer by Assaf Shocher

import numpy as np
from scipy.ndimage import filters, measurements, interpolation
from skimage import color
from math import pi
#from SinGAN.functions import torch2uint8, np2torch
import torch

# y=(x+1)/2 ,0<=y<=1 把输出限制在[0,1]之间
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# y=(x-0.5)*2 ,-1<=y<=1 把输出限制在[-1,1]之间
def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)

# 加载到设备，选择cpu/gpu
def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t

# 将数据类型从np变为torch类型
def np2torch(x,opt):
    # 改变数组索引
    if opt.nc_im == 3: # 图像通道数
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not (opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    #x = x.type(torch.cuda.FloatTensor)
    x = norm(x)
    return x

# 将数据类型从torch变为uint8类型
def torch2uint8(x):
    x = x[0,:,:,:]
    # 将tensor的维度换位（改变索引）
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

# 缩放图像尺寸
def imresize(im,scale,opt):
    #s = im.shape
    im = torch2uint8(im)
    im = imresize_in(im, scale_factor=scale)
    im = np2torch(im,opt)
    #im = im[:, :, 0:int(scale * s[2]), 0:int(scale * s[3])]
    return im

# 调整图像大小
def imresize_to_shape(im,output_shape,opt):
    #s = im.shape
    im = torch2uint8(im)
    im = imresize_in(im, output_shape=output_shape)
    im = np2torch(im,opt)
    #im = im[:, :, 0:int(scale * s[2]), 0:int(scale * s[3])]
    return im

def imresize_in(im, scale_factor=None, output_shape=None, kernel=None, antialiasing=True, kernel_shift_flag=False):
    # 首先通过从输出形状导出比例来标准化值并填充缺失的参数（如果需要），反之亦然
    scale_factor, output_shape = fix_scale_and_size(im.shape, output_shape, scale_factor)
    # 对于给定的数字内核情况，只需进行卷积和子采样（仅缩小）
    if type(kernel) == np.ndarray and scale_factor[0] <= 1:
        return numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag)
    # 选择插值方法
    method, kernel_width = {
        "cubic": (cubic, 4.0),
        "lanczos2": (lanczos2, 4.0),
        "lanczos3": (lanczos3, 6.0),
        "box": (box, 1.0),
        "linear": (linear, 2.0),
        None: (cubic, 4.0)  # 默认插值方法
    }.get(kernel)
    # 抗锯齿在缩小图像时才使用
    antialiasing *= (scale_factor[0] < 1)
    # 根据每个维度的比例对维度索引进行排序
    sorted_dims = np.argsort(np.array(scale_factor)).tolist()
    # 迭代维度，计算每次在一个方向上调整大小的局部权重
    out_im = np.copy(im)
    for dim in sorted_dims:
        # 比例因子为1时无意义
        if scale_factor[dim] == 1.0:
            continue
        # 对于每个坐标（一个维度），计算输入图像中的哪些坐标影响其结果并将其值相乘得到权重
        weights, field_of_view = contributions(im.shape[dim], output_shape[dim], scale_factor[dim],
                                               method, kernel_width, antialiasing)
        # 计算一个维度上调整大小的结果
        out_im = resize_along_dim(out_im, dim, weights, field_of_view)
    return out_im

def fix_scale_and_size(input_shape, output_shape, scale_factor):
    # 首先将比例因子（如果给定）固定为函数期望的标准化（与输入维度数量相同大小的比例因子列表）
    if scale_factor is not None:
        # 如果scale-factor 是标量，定义为 2d
        if np.isscalar(scale_factor):
            scale_factor = [scale_factor, scale_factor]
        # 将比例因子扩展到输入的大小
        scale_factor = list(scale_factor)
        scale_factor.extend([1] * (len(input_shape) - len(scale_factor)))

    # 扩展原始输入图像到输入形状的大小
    if output_shape is not None:
        output_shape = list(np.uint(np.array(output_shape))) + list(input_shape[len(output_shape):])
    if scale_factor is None:
        scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)

    # 根据比例因子计算，处理丢失的输出形状
    if output_shape is None:
        output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))
    return scale_factor, output_shape

def contributions(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    # 此函数计算一组“过滤器”和一组稍后会使用到的 field_of_view
    # field_of_view 中的每个位置都会与匹配的过滤器相乘
    # 'weights'基于插值方法和子像素到像素中心的距离

    fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
    kernel_width *= 1.0 / scale if antialiasing else 1.0
    # 输出图像的坐标
    out_coordinates = np.arange(1, out_length+1)
    # 这些是输出坐标在输入图像坐标上的匹配位置
    match_coordinates = 1.0 * out_coordinates / scale + 0.5 * (1 - 1.0 / scale)
    # 过滤器的左边界
    left_boundary = np.floor(match_coordinates - kernel_width / 2)
    # 在每一侧添加一个像素
    expanded_kernel_width = np.ceil(kernel_width) + 2
    # 为每个输出位置确定一组field_of_view
    field_of_view = np.squeeze(np.uint(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1))
    # 为视野中的每个像素分配权重
    weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1)
    # 权重归一化
    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights == 0] = 1.0
    weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)
    # 使用镜像结构进行边界填充
    mirror = np.uint(np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))))
    field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]
    # 摆脱零权重的权重和像素位置
    non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
    weights = np.squeeze(weights[:, non_zero_out_pixels])
    field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

    return weights, field_of_view

def resize_along_dim(im, dim, weights, field_of_view):
    # 每一维度分开插值，交换索引，对维度进行调换
    tmp_im = np.swapaxes(im, dim, 0)
    # 将单一维度添加到权重矩阵中，就可以将它与张量相乘
    weights = np.reshape(weights.T, list(weights.T.shape) + (np.ndim(im) - 1) * [1])
    # tmp_im[field_of_view.T] 是 image_dims+1 阶的张量
    tmp_out_im = np.sum(tmp_im[field_of_view.T] * weights, axis=0)
    # 维度调换回来
    return np.swapaxes(tmp_out_im, dim, 0)

def numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag):
    if kernel_shift_flag:
        kernel = kernel_shift(kernel, scale_factor)
    # 做相关运算
    out_im = np.zeros_like(im)
    for channel in range(np.ndim(im)):
        out_im[:, :, channel] = filters.correlate(im[:, :, channel], kernel)
    # 采样
    return out_im[np.round(np.linspace(0, im.shape[0] - 1 / scale_factor[0], output_shape[0])).astype(int)[:, None],
                  np.round(np.linspace(0, im.shape[1] - 1 / scale_factor[1], output_shape[1])).astype(int), :]

def kernel_shift(kernel, sf):
    # 移动核的两个原因：
    # 1. 质心不在核的中心。
    # 2. 使左上角像素对应于sfXsf的中间像素。 默认情况下，奇数大小位于第一个像素的中间，而偶数大小的内核位于第一个第一个像素的左上角。
    # 如果两个条件都满足，测试方法：输入图像在插值时（常规双三次）与地面实况完全对齐。

    # 计算质心
    current_center_of_mass = measurements.center_of_mass(kernel)
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (sf - (kernel.shape[0] % 2))
    # 移位向量
    shift_vec = wanted_center_of_mass - current_center_of_mass
    # 先填充，不会因移位丢失信息
    kernel = np.pad(kernel, np.int(np.ceil(np.max(shift_vec))) + 1, 'constant')
    return interpolation.shift(kernel, shift_vec)


# 下面函数是插值方法， x表示距左边像素中心的距离
# 三次样条插值
def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5*absx3 - 2.5*absx2 + 1) * (absx <= 1) +
            (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * ((1 < absx) & (absx <= 2)))

# lanczos(a=2)插值
def lanczos2(x):
    return (((np.sin(pi*x) * np.sin(pi*x/2) + np.finfo(np.float32).eps) /
             ((pi**2 * x**2 / 2) + np.finfo(np.float32).eps))
            * (abs(x) < 2))
# 分段插值
def box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0

# lanczos(a=3)插值
def lanczos3(x):
    return (((np.sin(pi*x) * np.sin(pi*x/3) + np.finfo(np.float32).eps) /
            ((pi**2 * x**2 / 3) + np.finfo(np.float32).eps))
            * (abs(x) < 3))

# 线性插值
def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))
