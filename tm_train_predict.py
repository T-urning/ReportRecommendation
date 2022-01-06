# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
sequence labeling
"""
import ast
import os
import json
import pdb
import warnings
import random
import shutil
import argparse
from functools import partial
from collections import defaultdict
from loguru import logger
from typing import Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
from sklearn.metrics import precision_recall_fscore_support

#from paddlenlp.metrics import ChunkEvaluator
from utils.collate_from_paddlenlp import Stack, Tuple, Pad
from utils.utils import (write_by_lines, get_entities,
                seed_everywhere, TrainingRecord, EarlyStopping, 
                ChunkEvaluator, create_folders_if_not_exist)

from utils.losses import FocalLoss

from data_prepare import (TextMatchDataset, tm_convert_example_to_feature)

from models.baseline_models import BiLSTM, BiLSTMCRF, DMCNN
from models.bert_based_models import BertForKE, BertCrfWithContraintForKE, BertCrfForKE, BertForTM

warnings.filterwarnings('ignore')

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=50, help="Number of epoches for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--tag_path", type=str, default=None, help="tag set path")
parser.add_argument("--vocab_path", type=str, default=None, help="vocab path")
parser.add_argument("--model_name", type=str, default='bert', help="model name")
parser.add_argument("--data_path", type=str, default='./data', help="train data")
parser.add_argument("--do_train", type=ast.literal_eval, default=False, help="do train")
parser.add_argument("--do_test", type=ast.literal_eval, default=False, help="do test")
parser.add_argument("--do_predict", type=ast.literal_eval, default=False, help="do predict")
parser.add_argument("--test", type=ast.literal_eval, default=False, help="do badcase analysis")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--max_seq_len", type=int, default=64, help="Number of words of the longest seqence.")
parser.add_argument("--valid_step", type=int, default=300, help="validation step")
parser.add_argument("--skip_step", type=int, default=50, help="skip step")
parser.add_argument("--train_batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--eval_batch_size", type=int, default=32, help="Total examples' number in batch for validating.")
parser.add_argument("--init_ckpt", type=str, default=None, help="already pretraining model checkpoint")
parser.add_argument("--output_dir", default="outputs", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--predict_save_path", type=str, default=None, help="predict data save path")
parser.add_argument("--seed", type=int, default=24, help="random seed for initialization")
parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use, 0 for CPU.")
args = parser.parse_args()
# yapf: enable.

@dataclass  
class ModelTestOutput(object):
    precision: float = None
    recall: float = None
    f1: float = None
    description: Optional[str] = None
@dataclass
class ModelTrainOutput(object):
    epochs: int = None
    dev_prec: float = None
    dev_recall: float = None
    dev_f1: float = None
@dataclass
class HyperparasForModelSearch(ModelTrainOutput, ModelTestOutput):
    model_name: str = None
    learning_rate : float = None
    gamma: int = None # for Focal loss
    weight: tuple = None # for Focal loss
    epsilon: float = 0.5
    weight_decay: float = None
    batch_size: int = None
    others: Optional[str] = None



def do_train(model, train_loader, dev_loader, criterion, optimizer, scheduler):
   
    logger.info("============start train==========")
    num_training_steps = len(train_loader) * args.num_epoch

    model_save_root = os.path.join(args.checkpoint_dir, "epoch_{}")
    early_stopping = EarlyStopping(patience=20, metric='f1', trace_func=logger.info)
    training_record = TrainingRecord(os.path.join(args.record_dir, f'{args.model_name}_{args.pretrained_model}_training'), logger.info)
    train_output = ModelTrainOutput()
    step, best_f1 = 0, 0.0
    model.train()
    for epoch in range(args.num_epoch):
        for idx, (input_ids, token_type_ids, seq_lens, labels) in enumerate(train_loader):
            input_ids, labels, token_type_ids = (
                torch.LongTensor(input_ids).to(args.device), 
                torch.LongTensor(labels).to(args.device),
                torch.LongTensor(token_type_ids).to(args.device)
            )
            
            masks = [[1 if i < length else 0 for i in range(input_ids.size(1))] for length in seq_lens]
            masks = torch.FloatTensor(masks).to(args.device)
            optimizer.zero_grad()
            result = model(input_ids, attention_mask=masks) #TODO 加上 token_type_ids
            if result.loss is not None:
                loss = result.loss
            else:
                logits = result.logits.view([-1, result.logits.size(-1)])
                loss = torch.mean(criterion(logits, labels.view([-1])))
            
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            loss_item = loss.item()
            if step > 0 and step % args.skip_step == 0:
                logger.info(f'train epoch: {epoch+1} - step: {step} (total: {num_training_steps}) - loss: {loss_item:.6f}')
            if step > 0 and step % args.valid_step == 0:
                p, r, f1, loss_avg = evaluate(model, criterion, dev_loader)
                
                # adjust the learning rate
                scheduler.step(loss_avg)
                # 记录训练时的指标输出
                if f1 > best_f1:
                    best_f1 = f1
                    train_output.dev_recall, train_output.dev_prec, train_output.dev_f1, train_output.epochs = (
                        r, p, f1, epoch+1
                    )
                training_record.add_record(**{'loss': loss_avg, 'precision': p, 'recall': r, 'f1': f1})
                
                logger.info(f'dev step: {step} - loss: {loss_avg:.5f}, precision: {p:.5f}, recall: {r:.5f}, ' \
                        f'f1: {f1:.5f}， current best f1: {best_f1:.5f}')
                
                # early_stopping 会负责保存最优模型
                model_save_path = model_save_root.format(epoch)
                
                if early_stopping(f1, model, model_save_path):
                    logger.info("Stopped training due to the loss didn't decrease for a long time.")
                    break
            step += 1

        if early_stopping.stop():
            logger.info('Stopped training.')
            break   # 若是触发 early_stopping 则会执行这里的 break 以跳出最外层 for 训练结束训练

    training_record.save_as_imgs()
    logger.info("=====training complete=====")
    return train_output

@torch.no_grad()
def evaluate(model, criterion, data_loader):
    """evaluate"""
    model.eval()

    losses = []
    gold_labels = []
    pred_labels = []
    total_iters = len(data_loader.dataset) // data_loader.batch_size + 1
    for idx, (input_ids, token_type_ids, seq_lens, labels) in tqdm(enumerate(data_loader), total=total_iters):
        input_ids, labels, token_type_ids = (
            torch.LongTensor(input_ids).to(args.device), 
            torch.LongTensor(labels).to(args.device),
            torch.LongTensor(token_type_ids).to(args.device)
        )
        masks = [[1 if i < length else 0 for i in range(input_ids.size(1))] for length in seq_lens]
        masks = torch.FloatTensor(masks).to(args.device)
        result = model(input_ids, attention_mask=masks) #TODO 加上 token_type_ids
        if result.loss is not None:
            loss = result.loss
        else:
            logits = result.logits.view([-1, result.logits.size(-1)])
            loss = torch.mean(criterion(logits, labels.view([-1])))

        losses.append(loss.item())
        if result.predictions is not None:
            preds = result.predictions
        else:
            preds = torch.argmax(result.logits, axis=-1)
        gold_labels.extend(labels.cpu().numpy().flatten())
        pred_labels.extend(preds.cpu().numpy().flatten())
    avg_loss = np.mean(losses)
    precision, recall, f1_score, _ = precision_recall_fscore_support(gold_labels, pred_labels, average='binary')
    model.train()

    return precision, recall, f1_score, avg_loss


@torch.no_grad()
def do_predict(model, raw_texts, predict_loader):
    model.to(args.device).eval()
    predict_results = []
    example_idx = 0
    total_iters = len(predict_loader.dataset) // predict_loader.batch_size + 1
    for batch_idx, (input_ids, token_type_ids, seq_lens) in tqdm(enumerate(predict_loader), total=total_iters):

        input_ids, token_type_ids = (
            torch.LongTensor(input_ids).to(args.device), 
            torch.LongTensor(token_type_ids).to(args.device)
        )
        masks = [[1 if i < length else 0 for i in range(input_ids.size(1))] for length in seq_lens]
        masks = torch.FloatTensor(masks).to(args.device)
        result = model(input_ids, masks, token_type_ids=token_type_ids)
        
        if result.predictions is not None:
            preds = result.predictions
        else:
            preds = torch.exp(result.logits[:, -1])
            # preds = result.logits[:, -1] # torch.argmax(result.logits, axis=-1)

        predictions = preds.cpu().numpy()
        
        for pred in predictions:
            
            predict_results.append({
                'id': example_idx,
                'text': raw_texts[example_idx],
                'prob': str(pred.item())
            })
            example_idx += 1
    assert len(predict_results) == example_idx
    return predict_results



@torch.no_grad()
def badcase_analysis(model, save_path, data_loader, metric, desc=''):
    model.to(args.device).eval()
    metric.reset()
    writer = open(f'{save_path}/{args.model_name}_{args.pretrained_model}_{desc}_badcase.txt', mode='w', encoding='utf-8')
    bound_false_total, false_total = 0, 0
    right_for_event_num, total_for_event_num = defaultdict(int), defaultdict(int)
    for idx, (input_ids, token_type_ids, seq_lens, labels) in enumerate(data_loader):
        input_ids, labels, token_type_ids = (
            torch.LongTensor(input_ids).to(args.device), 
            torch.LongTensor(labels).to(args.device),
            torch.LongTensor(token_type_ids).to(args.device))
        masks = [[1 if i < length else 0 for i in range(input_ids.size(1))] for length in seq_lens]
        masks = torch.FloatTensor(masks).to(args.device)
        result = model(input_ids, masks, token_type_ids=token_type_ids, labels=labels, is_predict=True)
        # logits = result.logits.view([-1, len(args.label_map)])
        if result.predictions is not None:
            preds = result.predictions
        else:
            preds = torch.argmax(result.logits, axis=-1)
        
        n_infer, n_label, n_correct = metric.compute(None, seq_lens, preds.cpu(), labels.cpu())
        n_false = (n_infer - n_correct).item()
        metric.update(n_infer, n_label, n_correct)
        precision, recall, f1_score = metric.accumulate()
        check_result = metric.check(None, seq_lens, preds.cpu(), labels.cpu())
        event_num = n_label.item()
        if event_num >= 4:
            pass
        if check_result:
            #logger.warning(f"{i+1}th example predicted wrong.")
            unpad_predictions, unpad_labels, n_bound_false = check_result
            
            writer.write(f"{idx+1}:\t边界错 {n_bound_false} 个，类别错 {n_false - n_bound_false} 个\n\n")
            writer.write(f"Gold:\t{unpad_labels[0]}\n")
            writer.write(f"Pred:\t{unpad_predictions[0]}\n\n")
            writer.flush()
            bound_false_total += n_bound_false
            false_total += n_false
        else:
            right_for_event_num[event_num] += 1
            assert n_false == 0
        total_for_event_num[event_num] += 1

        

    ratio_dict = {}
    for key in total_for_event_num.keys():
        ratio_dict[key] = right_for_event_num[key] / total_for_event_num[key]
    writer.close()
    logger.warning(f"{'*'*50}")
    logger.warning(f"按事件数正确：{dict(right_for_event_num)}，总数：{dict(total_for_event_num)}，占比：{ratio_dict}")
    logger.warning(f"共预测 {metric.num_infer_chunks} 个，真实有 {metric.num_label_chunks} 个， 预测正确的有 {metric.num_correct_chunks} 个")
    class_false_total = false_total - bound_false_total
    description = f"关键词共错 {false_total} 个，其中分类错 {class_false_total} 个，占比 {class_false_total/false_total:.5f}"
    logger.warning(description)
    logger.warning(f"precision: {precision:.5f}, recall: {recall:.5f}, f1: {f1_score:.5f}")
    logger.warning(f"{'*'*50}")

    return ModelTestOutput(
        precision=precision,
        recall=recall,
        f1=f1_score,
        description=description
    )


def main(model_map):
    
    pretrained_model_path = f"./pretrained_models/{args.pretrained_model}/"
    
    # 创建不存在的文件夹
    checkpoint_root = os.path.join(args.output_dir, 'checkpoints')
    model_root = os.path.join(checkpoint_root, args.model_name)
    args.checkpoint_dir = model_root #os.path.join(model_root, args.pretrained_model)
    args.record_dir = os.path.join(args.output_dir, 'record_as_imgs')
    log_save_path = os.path.join(args.output_dir, 'logs')
    badcase_save_path = os.path.join(args.output_dir, 'badcases')
    prediction_save_path = os.path.join(args.output_dir, 'predictions')
    
    create_folders_if_not_exist(
        args.output_dir,
        checkpoint_root,
        model_root,
        args.checkpoint_dir, 
        args.record_dir, 
        log_save_path, 
        badcase_save_path,
        prediction_save_path
    )

    handler_id = logger.add(
        f"{log_save_path}/{args.model_name}_{args.pretrained_model}_training.log", 
        level="INFO", rotation="10 MB", encoding='utf-8', mode='a'
    )

    model_path = pretrained_model_path
    if args.do_test or args.do_predict:
        model_path = os.path.join(args.checkpoint_dir, f'epoch_{args.test_epoch}')
    
    model_config_path = os.path.join(model_path, 'config.json')
    vocab_path = os.path.join(model_path, 'vocab.txt')
    try:
        if not os.path.exists(model_config_path):
            shutil.copyfile(os.path.join(pretrained_model_path, 'config.json'), model_config_path)
        if not os.path.exists(vocab_path):
            shutil.copyfile(os.path.join(pretrained_model_path, 'vocab.txt'), vocab_path)
    except OSError as oe:
        logger.warning(f'config.json or vocab.txt file are not exist in {pretrained_model_path}')


    model = model_map[args.model_name].from_pretrained(
                model_path, **{'num_labels': 2, 'max_seq_len': args.max_seq_len})
    model.to(args.device)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    trans_func = partial(
        tm_convert_example_to_feature,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        is_test=args.do_predict)

    #metric = ChunkEvaluator(id2label_dict=args.id2label, suffix=False)
    if not args.do_predict:
    
        # loss function
        loss_function = FocalLoss(gamma=args.gamma, weight=None)
        
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]), # input ids
            Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]), # token type ids
            Stack(), # sequence lens
            Stack() # labels
        ): fn(list(map(trans_func, samples)))

        # analysis
        if args.do_test:
            test_path = os.path.join(args.data_path, 'test.tsv')
            test_ds = TextMatchDataset(test_path)
            test_loader = DataLoader(
                dataset=test_ds,
                batch_size=args.eval_batch_size,
                collate_fn=batchify_fn
            )
            precision, recall, f1_score, avg_loss = evaluate(model, loss_function, test_loader)
            test_output = ModelTestOutput(
                precision=precision,
                recall=recall,
                f1=f1_score,
                description=''
            )
            logger.remove(handler_id)
            return test_output

        train_path = os.path.join(args.data_path, 'train.tsv')
        train_ds = TextMatchDataset(train_path)
        
        train_size = int(0.88 * len(train_ds))
        dev_size = len(train_ds) - train_size
        # 指定 generator 固定住 train dataset 和 dev dataset
        train_ds, dev_ds = random_split(train_ds, [train_size, dev_size], generator=torch.Generator().manual_seed(24))

        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=batchify_fn)
            
        dev_loader = DataLoader(
            dataset=dev_ds,
            batch_size=args.eval_batch_size,
            collate_fn=batchify_fn)
        # Defines optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=1e-8
        )
        
        # Learning rate schedular: reduce learning rate when a metric has stopped improving
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            patience=8,
            factor=0.1
        )
        train_output = do_train(model, train_loader, dev_loader, loss_function, optimizer, scheduler)
        return train_output


    else:

        raw_texts = []
        test_path = os.path.join(args.data_path, 'predict.tsv')
        with open(test_path, 'r', encoding='utf-8') as f:
            # skip the head line
            next(f)
            for line in f:
                words_1, words_2 = line.strip('\n').split('\t')[:2]
                raw_texts.append([''.join(words_1.split('\002')), ''.join(words_2.split('\002'))])

        test_ds = TextMatchDataset(test_path)
        predict_batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]), # input ids
            Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]), # token type ids
            Stack() # sequence lens
        ): fn(list(map(trans_func, samples)))

        test_loader = DataLoader(
            dataset=test_ds,
            batch_size=args.eval_batch_size,
            collate_fn=predict_batchify_fn
        )
        predict_results = do_predict(model, raw_texts, test_loader)

        predict_results = [json.dumps(r, ensure_ascii=False) for r in predict_results]
        prediction_save_file = os.path.join(prediction_save_path, f'{args.model_name}_{args.pretrained_model}_{args.test_epoch}_prediction_text_matching.jl') 
        write_by_lines(prediction_save_file, predict_results)

        print('Prediciting is done.')
    
    logger.remove(handler_id) # 移除 logger 之前的 handler

if __name__ == '__main__':
    
    seed_everywhere()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_map = {
        'bert': BertForTM
    }
    lr_map = {
        'bert': 2e-5
    }

    
    #args.model_name = 'bert'
    args.data_path = 'data/text_match/' # 'data/journals/'
    args.output_dir = 'outputs/text_match'
    args.pretrained_model = 'bert_wwm_ext'
    
    searched_times = 0
    record_writer = open('model_search_record_for_text_match_baselines.txt', mode='a', encoding='utf-8')
    for lr in [1e-5, 2e-5, 5e-5, 7e-5, 9e-5]:
        for time in range(1):
        
            args.model_name = 'bert'
            args.learning_rate = lr #lr_map[args.model_name]
            args.epsilon = None

            #args.loss_function_name = 'cross_entropy_loss'
            # for focal loss
            args.gamma = 0
            args.weight = (1,1)
            
            # args.do_train = True
            # args.do_predict = True

            args.do_predict = True
            args.test_epoch = 2

            if args.do_test:
                logger.warning(f"{'*'*50}")
                logger.warning(f"{args.model_name} testing")
                logger.warning(f"{'*'*50}")
            elif args.do_predict:
                logger.warning(f"{'*'*50}")
                logger.warning(f"{args.model_name} predicting")
                logger.warning(f"{'*'*50}")
            else:
                logger.warning(f"{'*'*50}")
                logger.warning(f"{args.model_name} training")
                logger.warning(f"{'*'*50}")

            output = main(model_map)

            if not args.do_predict:

                if args.do_test:
                    logger.info(f"model test output: {output}")
                else:
                    logger.info(f"model train output: {output}")


                hyparas_for_model = HyperparasForModelSearch(
                    **output.__dict__,
                    model_name=args.model_name,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    batch_size=args.train_batch_size,
                    gamma=args.gamma,
                    weight=args.weight,
                    epsilon=args.epsilon
                )
                hyparas = hyparas_for_model.__dict__
                if searched_times == 0:
                    record_writer.write('\t'.join(hyparas.keys()) + '\n')

                hypara_values = [str(value) if not isinstance(value, float) else str(round(value, 5)) for value in hyparas.values()]
                record_writer.write('\t'.join(hypara_values) + '\n')
                record_writer.flush()

                searched_times += 1

    record_writer.close()
        
    
