#============================================================
# Create Time:			2021-07-06 18:29:59
# Last modify:			2021-07-06 18:49:39
# Writer:				Wenhao	1795902848@qq.com
# File Name:			my_single_fid_score.py
# File Type:			PY Source File
# Tool:					Mac -- vim & python
# Information:			计算两个图片的fid距离。
#============================================================
import argparse
from inception import InceptionV3
from sifid_score import calculate_frechet_distance
from sifid_score import calculate_activation_statistics

def calculate_signle_fid_score(path1, path2, cuda=True, batch_size=1, dims=64):
    """
    @createTime: 2021-07-06 18:46:56
    @arg: path1 ：图片1的路径，可以是相对或绝对路径。
    @arg: path2 ：图片2的路径，可以是相对或绝对路径。
    @arg: cuda=True 
    @arg: batch_size=1 
    @arg: dims=64 
    @rtn: 
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    m1, s1 = calculate_activation_statistics([path1], model, batch_size, dims, cuda)
    m2, s2 = calculate_activation_statistics([path2], model, batch_size, dims, cuda)
    score = calculate_frechet_distance(m1, s1, m2, s2)

    return score

def build_args():
    parser = argparse.ArgumentParser('计算两个图片的fid距离。')
    parser.add_argument(
        '--path2real', type=str, required=False,
        default='../Input/Images/my_trees.png',
        help='原始图片的路径。'
    )
    parser.add_argument(
        '--path2fake', type=str, required=False,
        default='../Output/RandomSamples/my_trees/gen_start_scale=0/0.png',
        help='生成的图片的路径。'
    )

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = build_args()
    score = calculate_signle_fid_score(opt.path2real, opt.path2fake)
    print('  `%s`\n和`%s`\n两个图片的FID距离为：%.4f'%(
        opt.path2real, opt.path2fake, score
    ))
