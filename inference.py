import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
from resnet import resnet as caffe_resnet
import argparse
from PIL import Image
import numpy as np
from torchvision import transforms
import config
import utils

import json


def attn_hook_function(module, inputs, outputs):
    global img
    input_images = inputs[0]        # [n, c, h, w]
    n, c, h, w = input_images.shape

    attention_weight = outputs[-1]  # [n, g, s=h*w]
    n, g, s = attention_weight.shape
    attention_weight = attention_weight.reshape(n, g, h, w)
    attention_weight = F.interpolate(attention_weight, (img.size[1], img.size[0]), mode='bicubic')

    assert n == 1
    attention_weight = attention_weight.squeeze(0)  # remove batch_size (g, h, w)
    list_attention_imgs = []
    for i in range(attention_weight.size(0)):
        weight = attention_weight[i].cpu()      # h, w
        weight_image_np = weight.numpy()
        weight_image_np = np.expand_dims(weight_image_np, -1)   # h, w => h, w, 1
        image_np = np.array(img)                                # h, w, 3
        attention_image_np = image_np * weight_image_np * 255.
        attention_image_np = np.clip(attention_image_np, 0, 255)
        attention_image = Image.fromarray(attention_image_np.astype(np.uint8))
        attention_image.save(f'./visualize_attention/{i}.png')
        # img.save(f'./visualize_attention/{i}.png')
        # Image.blend(img, weight_image, )
        # list_attention_imgs.append(weight_image)





class ApplyAttention(nn.Module):
    def __init__(self):
        super(ApplyAttention, self).__init__()

    def forward(self, input, attention):
        """ Apply any number of attention maps over the input. """
        n, c = input.size()[:2]
        glimpses = attention.size(1)

        # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
        input = input.view(n, 1, c, -1) # [n, 1, c, s] s = 14*14
        attention = attention.view(n, glimpses, -1)
        attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]
        weighted = attention * input # [n, g, v, s]
        weighted_mean = weighted.sum(dim=-1) # [n, g, v]
        return weighted_mean, attention.squeeze(2)


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled


class ImageProcessor(nn.Module):
    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.model = caffe_resnet.resnet152(pretrained=True) # resnet34

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer

class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1)
        self.features = lstm_features

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len.cpu(), batch_first=True)
        _, (_, c) = self.lstm(packed)
        return c.squeeze(0)


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x))
        return x


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))


class Net(nn.Module):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, embedding_tokens):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        glimpses = 2

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=question_features,
            drop=0.5,
        )
        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=2,
            drop=0.5,
        )
        self.classifier = Classifier(
            in_features=glimpses * vision_features + question_features,
            mid_features=1024,
            out_features=config.max_answers,
            drop=0.5,
        )

        self.apply_attention = ApplyAttention()

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_len):
        q = self.text(q, q_len)

        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        a = self.attention(v, q)  ####
        v = self.apply_attention(v, a)[0]
        v = v.view(v.size(0), -1)

        combined = torch.cat([v, q], dim=1)
        answer = self.classifier(combined)
        answer = F.softmax(answer, dim=1)
        return answer

def main():
    global img
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='path to image.')
    parser.add_argument('question', help='question about image')
    args = parser.parse_args()

    # Image preprocess
    img = Image.open(args.input_dir).convert('RGB')
    transform = utils.get_transform(config.image_size, config.central_fraction)
    v = transform(img)

    net = ImageProcessor()
    net.eval()

    v = v.unsqueeze(dim=0)
    with torch.no_grad():
        v = net(v)

    # Question preprocess
    q = args.question
    q = q.lower()[:-1]
    q = q.split(' ')
    q_len = torch.tensor([len(q)], dtype=torch.long)

    max_question_length = 23
    with open(config.vocabulary_path, 'r') as fd:
        vocab_json = json.load(fd)

    vec = torch.zeros(max_question_length).long()
    
    token_to_index = vocab_json['question']
    
    for i, token in enumerate(q):
        index = token_to_index.get(token, 0)
        vec[i] = index
    
    vec = vec.unsqueeze(dim=0)

    num_tokens = len(token_to_index) + 1

    log = torch.load('2017-08-04_00.55.19.pth', map_location='cpu')
    net = torch.nn.DataParallel(Net(num_tokens))
    net.load_state_dict(log['weights'])
    net.eval()
    net.module.apply_attention.register_forward_hook(attn_hook_function)
    with torch.no_grad():
        out = net(v, vec, q_len)
    conf, ans = out.topk(k=5, dim=1)
    conf, ans = conf.tolist()[0], ans.tolist()[0]

    for c, a in zip(conf, ans):
        print(get_key(a, vocab_json['answer']), c)

if __name__ == '__main__':
    main()

    
