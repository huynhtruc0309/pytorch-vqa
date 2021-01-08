import sys
import os.path
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import config
import data
import model
import utils

def run(net, loader):
    """ Run an epoch over the given loader """
    net.eval()
    answ = []
    idxs = []
    accs = []
    log_softmax = nn.LogSoftmax().cuda()

    for v, q, a, idx, q_len in loader:
        var_params = {
            'volatile': True,
            'requires_grad': False,
        }
        v = Variable(v.cuda(non_blocking=True), **var_params)
        q = Variable(q.cuda(non_blocking=True), **var_params)
        a = Variable(a.cuda(non_blocking=True), **var_params)
        q_len = Variable(q_len.cuda(non_blocking=True), **var_params)

        out = net(v, q, q_len)
        nll = -log_softmax(out)
        loss = (nll * a / 10).sum(dim=1).mean()
        acc = utils.batch_accuracy(out.data, a.data).cpu()

        _, answer = out.data.cpu().max(dim=1)
        answ.append(answer.view(-1))
        accs.append(acc.view(-1))
        idxs.append(idx.view(-1).clone())

    answ = list(torch.cat(answ, dim=0))
    accs = list(torch.cat(accs, dim=0))
    idxs = list(torch.cat(idxs, dim=0))
    return answ, accs, idxs

def main():
    path = sys.argv[1]
    test_loader = data.get_loader(test=True)

    log = torch.load(path)
    tokens = len(log['vocab']['question']) + 1

    net = torch.nn.DataParallel(model.Net(tokens)).cuda()
    net.load_state_dict(log['weights'])

    r = run(net, test_loader)
    print('answers', r[0], 'accuracies', r[1], 'idx', r[2])


if __name__ == '__main__':
    main()
