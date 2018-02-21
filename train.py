# _*_ coding:utf-8 _*_
import sys, os
import torch
from data import get_data
from model import PoetryModel
from torch import nn
from torch.autograd import Variable
import Config as cfg
import torch.utils.data as td

opt = cfg.Config()
def train(opt):
    data, word2ix, ix2word = get_data(opt)
    data = torch.from_numpy(data)
    dataloader = td.DataLoader(data,
                               batch_size=opt.batch_size,
                               shuffle=True,
                               num_workers=1)

    model = PoetryModel(len(word2ix), 128, 256)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()
    if opt.use_gpu:
        model.cuda()
        criterion.cuda()
    for epoch in range(opt.epoch):
        for step, data_ in enumerate(dataloader):
            data_ = data_.long().transpose(1, 0).contiguous()
            if opt.use_gpu: data_ = data_.cuda()
            optimizer.zero_grad()
            input_, target = Variable(data_[:-1, :]), Variable(data_[1:, :])
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()

            if (1 + step) % 10 == 0:
                print("current loss",loss.data)

        t.save(model.state_dict(), '%s_%s.pth' % (opt.model_prefix, epoch))

t.save(model.state_dict(),"checkpoints/final.pth")

if __name__ == '__main__':
    train(opt)



