import os
import time
import json
import torch
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

doc_topics = ['会议', '会见', '调研', '活动', '批示']
paragraph_topics = ['经济', '健康', '农业', '劳动力', '社会福利', '文化教育', '环境', '能源', '基建', '住房', '金融', '科技', '应急管理', '财税', '政府', '组织', '宣传', '政法', '纪检', '统战', '人大政协', '国防', '综合']

class CustomizedDataset(Dataset):
    def __init__(self, srcs, tgts, tokenizer):
        self.tokenizer = tokenizer
        srcs, tgts = list(srcs), list(tgts)
        try:
            assert len(srcs) == len(tgts)
        except:
            print("srcs:", len(srcs))
            print("tgts:", len(tgts))
            raise Exception("srcs and tgts must be of the same length")
        self.length = len(srcs)
        self.source_ids, self.source_mask, self.target_ids = self.tokenize(srcs, tgts)

    def tokenize(self, srcs, tgts):
        input_ids = self.tokenizer.batch_encode_plus(srcs, max_length=512, padding=True, truncation=True, return_tensors='pt')
        source_ids = input_ids.input_ids
        source_mask = input_ids.attention_mask
        target_ids = torch.LongTensor([[int(i)] for i in tgts])
        return source_ids, source_mask, target_ids
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return (
            self.source_ids[idx],
            self.source_mask[idx],
            self.target_ids[idx]
        )

def load_data(file_path):
    data = json.load(open(file_path, 'r'))
    data = ([x['content'] for x in data], [x['topic_id'] for x in data])
    return data

def load_inference_data(file_path):
    data = json.load(open(file_path, 'r'))
    data = ([x['content'] for x in data], [-1 for x in data])
    return data
    
def load_and_split_data(file_path):
    data = json.load(open(file_path, 'r'))
    random.shuffle(data)
    train_data = data[:int(len(data) * 0.8)]
    train_data = ([x['content'] for x in train_data], [x['topic_id'] for x in train_data])
    test_data = data[int(len(data) * 0.8):]
    test_data = ([x['content'] for x in test_data], [x['topic_id'] for x in test_data])
    return train_data, test_data

def prepare_data(args, data, tokenizer, training=True, distributed=False):
    srcs, tgts = data
    if len(set(tgts)) > args.num_labels:
        raise Exception("Number of labels {} exceeds the limit {}".format(len(set(tgts)), args.num_labels))
    elif len(set(tgts)) < args.num_labels and not args.generate:
        print("Number of labels {} is less than the limit {}".format(len(set(tgts)), args.num_labels))
    dataset = CustomizedDataset(srcs, tgts, tokenizer)
    if distributed and args.n_gpus > 1:
        data_sampler = DistributedSampler(dataset)
        data_loader = DataLoader(dataset, sampler=data_sampler, batch_size=args.batch_size)
    else:
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=training, pin_memory=True)
    return data_loader

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='training')
        >>> step = 2
        >>> pbar(step=step)
    '''
    def __init__(self, n_total,width=30,desc = 'Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current< self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')