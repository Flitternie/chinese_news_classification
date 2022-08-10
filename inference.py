import os
import sys
import time
import logging
import argparse
from unittest import load_tests

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from utils import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings

def validate(args, model, data, device):
    model.eval()
    model = model.module if hasattr(model, "module") else model
    
    with torch.no_grad():
        all_outputs = []
        all_targets = []
        for batch in tqdm(data, total=len(data)):
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask, labels = batch[0], batch[1], batch[2]  
            inputs = {
                "input_ids": source_ids.to(device),
                "attention_mask": source_mask.to(device),
                # "labels": labels.to(device),
            }
            outputs = model(**inputs).logits
            all_targets.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.argmax(axis=-1).cpu().numpy())

    assert len(all_targets) == len(all_outputs)
    accuracy = np.mean([1 if output == target else 0 for output, target in zip(all_outputs, all_targets)])
    logging.info('Accuracy: {}'.format(accuracy))
    return accuracy, all_outputs

def generate(args, model, data, device):
    model.eval()
    model = model.module if hasattr(model, "module") else model
    
    with torch.no_grad():
        all_outputs = []
        all_targets = []
        for batch in tqdm(data, total=len(data)):
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask, labels = batch[0], batch[1], batch[2]  
            inputs = {
                "input_ids": source_ids.to(device),
                "attention_mask": source_mask.to(device),
                # "labels": labels.to(device),
            }
            outputs = model(**inputs).logits
            all_targets.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.argmax(axis=-1).cpu().numpy())

    return all_outputs

def inference(args):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    logging.info("Create model.........")
    _, model_class, tokenizer_class = (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    logging.info("Create test_loader.........")
    test_data = load_data(args.input_dir) if not args.generate else load_inference_data(args.input_dir)
    test_loader = prepare_data(args, test_data, tokenizer=tokenizer, training=False, distributed=False) 
    test_srcs, test_tgts = test_data
    if args.generate:
        outputs = generate(args, model, test_loader, device)
        with open(os.path.join(args.output_dir, 'pred.txt'), 'w') as f:
            if 'doc' in args.input_dir:
                for i in range(len(outputs)):
                    f.write(str(doc_topics[int(outputs[i])]) + '\t' + str(test_srcs[i]) + '\n')
            elif 'para' in args.input_dir: 
                for i in range(len(outputs)):
                    f.write(str(paragraph_topics[int(outputs[i])]) + '\t' + str(test_srcs[i]) + '\n')
            else:
                for i in range(len(outputs)):
                    f.write(str(outputs[i]) + '\t' + str(test_srcs[i]) + '\n')
        
    else:
        accuracy, outputs = validate(args, model, test_loader, device)
        try:
            assert len(test_srcs) == len(test_tgts) == len(outputs)
        except AssertionError:
            logging.error("Length of test_srcs, test_tgts, outputs are {}, {} and {}".format(len(test_srcs), len(test_tgts), len(outputs)))
            raise AssertionError
        with open(os.path.join(args.output_dir, 'pred.txt'), 'w') as f:
            f.write("预测" + '\t' + "实际" + '\t' + "正文" '\n')
            if 'doc' in args.input_dir:
                for i in range(len(outputs)):
                    f.write(doc_topics[int(outputs[i])] + '\t' + doc_topics[int(test_tgts[i])] + '\t' + str(test_srcs[i]) + '\n')
            elif 'para' in args.input_dir: 
                for i in range(len(outputs)):
                    f.write(paragraph_topics[int(outputs[i])] + '\t' + paragraph_topics[int(test_tgts[i])] + '\t' + str(test_srcs[i]) + '\n')
            else:
                for i in range(len(outputs)):
                    f.write(str(outputs[i]) + '\t' + str(test_tgts[i]) + '\t' + str(test_srcs[i]) + '\n')
        with open(os.path.join(args.output_dir, 'error.txt'), 'w') as f:
            f.write("预测" + '\t' + "实际" + '\t' + "正文" '\n')
            if 'doc' in args.input_dir:
                for i in range(len(outputs)):
                    if outputs[i] != test_tgts[i]:
                        f.write(doc_topics[int(outputs[i])] + '\t' + doc_topics[int(test_tgts[i])] + '\t' + str(test_srcs[i]) + '\n')
            elif 'para' in args.input_dir: 
                for i in range(len(outputs)):
                    if outputs[i] != test_tgts[i]:
                        f.write(paragraph_topics[int(outputs[i])] + '\t' + paragraph_topics[int(test_tgts[i])] + '\t' + str(test_srcs[i]) + '\n')
            else:
                for i in range(len(outputs)):
                    if outputs[i] != test_tgts[i]:
                        f.write(str(outputs[i]) + '\t' + str(test_tgts[i]) + '\t' + str(test_srcs[i]) + '\n')
            

def main():
    parser = argparse.ArgumentParser()

    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_name_or_path', required = True)

    # training parameters
    parser.add_argument('--num_labels', default=5, choices=[len(doc_topics), len(paragraph_topics)], type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--generate', action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    fileHandler = logging.FileHandler(os.path.join(args.output_dir, '{}.predict.log'.format(time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))

    seed_everything(args.seed)
    inference(args)


if __name__ == '__main__':
    main()

