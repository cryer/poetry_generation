

class Config(object):
    data_path = 'data/'  # 诗歌的文本文件存放路径
    pickle_path = 'data/tang.npz'  # 预处理好的二进制文件
    lr = 1e-3
    use_gpu = True
    epoch = 20
    batch_size = 128
    max_gen_len = 200
    model_path = 'checkpoints/final.pth'
    # prefix_words = '细雨鱼儿出，微风燕子斜。'
    prefix_words = '庄生晓梦迷蝴蝶，望帝春心托杜鹃。'
    start_words = '如果能重来'
    acrostic = False  # 是否是藏头诗
    model_prefix = 'checkpoints/cpkt'