import os
import sys
import time
import logging
import argparse

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from misc import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings
warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query

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

def inference(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info("Create model.........")
    _, model_class, tokenizer_class = (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    logging.info("Create test_loader.........")
    _, test_data = load_and_split_data(args.input_dir)
    test_loader = prepare_data(args, test_data, tokenizer=tokenizer, training=False, distributed=False) 
    accuracy, outputs = validate(args, model, test_loader, device)

    test_srcs, test_tgts = test_data
    with open(os.path.join(args.output_dir, 'pred.txt'), 'w') as f:
        if 'doc' in args.input_dir:
            for i in range(len(outputs)):
                f.write(doc_topics[int(outputs[i]) + 1] + '\t' + doc_topics[int(test_tgts[i]) + 1] + '\t' + str(test_srcs[i]) + '\n')
        elif 'para' in args.input_dir: 
            for i in range(len(outputs)):
                f.write(paragraph_topics[int(outputs[i]) + 1] + '\t' + paragraph_topics[int(test_tgts[i]) + 1] + '\t' + str(test_srcs[i]) + '\n')
        else:
            for i in range(len(outputs)):
                f.write(str(outputs[i]) + '\t' + str(test_tgts[i]) + '\t' + str(test_srcs[i]) + '\n')
            

def main():
    parser = argparse.ArgumentParser()

    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_name_or_path', required = True)

    # training parameters
    parser.add_argument('--num_labels', default=5, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')

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

