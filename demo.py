# _*_ coding:utf-8 _*_
import sys, os
import torch
from data import get_data
from model import PoetryModel
from torch.autograd import Variable
import numpy as np
import Config as cfg

opt = cfg.Config()

def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    results = list(start_words)
    start_word_len = len(start_words)
    input = Variable(torch.Tensor([word2ix['<START>']]).view(1, 1).long())
    if opt.use_gpu: input = input.cuda()
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = Variable(input.data.new([word2ix[word]])).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)

        if i < start_word_len:
            w = results[i]
            input = Variable(input.data.new([word2ix[w]])).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0]
            w = ix2word[top_index]
            results.append(w)
            input = Variable(input.data.new([top_index])).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results

def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    results = []
    start_word_len = len(start_words)
    input = Variable(torch.Tensor([word2ix['<START>']]).view(1, 1).long())
    if opt.use_gpu: input = input.cuda()
    hidden = None

    index = 0
    pre_word = '<START>'

    if prefix_words:
        for word in prefix_words:
            if word in word2ix:
                print("true..")
            else:
                print("false...please use Chinese input method")
                continue
            output, hidden = model(input, hidden)
            input = Variable(input.data.new([word2ix[word]])).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0]

        w = ix2word[top_index]

        if (pre_word in {u'。', u'！', '<START>'}):
            if index == start_word_len:
                break
            else:
                w = start_words[index]
                index += 1
                input = Variable(input.data.new([word2ix[w]])).view(1, 1)
        else:
            input = Variable(input.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w
    return results

def gen(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)
    data, word2ix, ix2word = get_data(opt)
    model = PoetryModel(len(word2ix), 128, 256)
    map_location = lambda s, l: s
    state_dict = torch.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)

    if opt.use_gpu:
        model.cuda()
    if sys.version_info.major == 3:
        if opt.start_words.isprintable():
            start_words = opt.start_words
            prefix_words = opt.prefix_words if opt.prefix_words else None
        else:
            start_words = opt.start_words.encode('ascii', 'surrogateescape').decode('utf8')
            prefix_words = opt.prefix_words.encode('ascii', 'surrogateescape').decode(
                'utf8') if opt.prefix_words else None
    else:
        start_words = opt.start_words.decode('utf8')
        prefix_words = opt.prefix_words.decode('utf8') if opt.prefix_words else None

    start_words = start_words.replace(',', u'，') \
        .replace('.', u'。') \
        .replace('?', u'？')

    gen_poetry = gen_acrostic if opt.acrostic else generate
    result = gen_poetry(model, start_words, ix2word, word2ix, prefix_words)
    print(''.join(result))

if __name__ == '__main__':
    import fire
    fire.Fire()