#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

"""
RTL Error Localization and Correction using GraphCodeBERT - Version 2
支持从datasets/rtl_training加载数据

修改内容:
1. 添加 load_rtl_dataset() 函数
2. 修改训练数据加载逻辑
3. 其他代码保持不变

使用示例:
python rtl_error_correction_v2.py \
    --do_train \
    --model_name_or_path microsoft/graphcodebert-base \
    --train_filename datasets/rtl_training/train.jsonl \
    --dev_filename datasets/rtl_training/valid.jsonl \
    --output_dir ./saved_models/rtl \
    --max_source_length 256 \
    --max_target_length 128 \
    --train_batch_size 16 \
    --num_train_epochs 10
"""

from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from error_correction_model import RTLErrorCorrectionModel
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from torch.optim import AdamW

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def extract_verilog_dataflow_mock(code):
    """
    Mock Verilog dataflow extraction until we can properly build tree-sitter-verilog.
    This creates a simple token-based DFG for testing purposes.
    """
    try:
        # Simple tokenization
        lines = code.split('\n')
        tokens = []
        for line_num, line in enumerate(lines):
            line_tokens = line.strip().split()
            for token_num, token in enumerate(line_tokens):
                if token and not token.startswith('//'):  # Skip comments
                    tokens.append(token.strip(';,()'))
        
        # Simple DFG: look for assignment patterns
        dfg = []
        for i, token in enumerate(tokens):
            if '=' in token or token == 'assign':
                # Create simple DFG edge
                if i > 0 and i < len(tokens) - 1:
                    left = tokens[i-1] if '=' in token else tokens[i+1]
                    right = tokens[i+1] if '=' in token else tokens[i+2] if i+2 < len(tokens) else ""
                    if left and right:
                        dfg.append((left, i-1, 'computedFrom', [right], [i+1]))
        
        return tokens, dfg
    except:
        return code.split(), []


# ============================================================
# 新增函数: 从datasets/rtl_training加载数据
# ============================================================

def load_rtl_dataset(filename):
    """
    Load RTL error correction dataset from JSON or JSONL file
    
    Supports both formats:
    - JSON: Array of objects (e.g., train.json)
    - JSONL: One JSON object per line (e.g., train.jsonl)
    
    Args:
        filename: Path to the dataset file
        
    Returns:
        List of examples, each with:
        - buggy_code: str - Code with defects
        - correct_code: str - Corrected code
        - comments: str - Description/comments
        - error_type: str - Type of error (optional)
        - id: int - Sample ID (optional)
    """
    examples = []
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset file not found: {filename}")
    
    logger.info(f"Loading RTL dataset from {filename}")
    
    try:
        if filename.endswith('.json'):
            # Load JSON array format
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("JSON file must contain an array of objects")
                
                for item in data:
                    examples.append({
                        'buggy_code': item['buggy_code'],
                        'correct_code': item['correct_code'],
                        'comments': item.get('comments', ''),
                        'error_type': item.get('error_type', 'unknown'),
                        'id': item.get('id', -1)
                    })
        
        elif filename.endswith('.jsonl'):
            # Load JSONL format (one JSON object per line)
            with open(filename, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        examples.append({
                            'buggy_code': item['buggy_code'],
                            'correct_code': item['correct_code'],
                            'comments': item.get('comments', ''),
                            'error_type': item.get('error_type', 'unknown'),
                            'id': item.get('id', -1)
                        })
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                        continue
        else:
            raise ValueError(f"Unsupported format: {filename}. Use .json or .jsonl")
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    
    logger.info(f"Successfully loaded {len(examples)} examples")
    
    # Print statistics
    error_types = {}
    for ex in examples:
        et = ex.get('error_type', 'unknown')
        error_types[et] = error_types.get(et, 0) + 1
    
    logger.info("Error type distribution:")
    for et, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {et}: {count} ({100*count/len(examples):.1f}%)")
    
    return examples


# ============================================================
# 原有函数保持不变
# ============================================================

def convert_examples_to_features(examples, tokenizer, args, stage="train"):
    """Convert examples to features for training/inference"""
    features = []
    
    for idx, example in enumerate(tqdm(examples, desc="Converting examples")):
        # Get buggy code, correct code, and comments
        buggy_code = example.get('buggy_code', '')
        correct_code = example.get('correct_code', '')
        comments = example.get('comments', '')
        
        # Extract dataflow from buggy code (mock for now)
        code_tokens, dfg = extract_verilog_dataflow_mock(buggy_code)
        
        # Combine source: comments + buggy code + DFG nodes
        source_tokens = []
        source_tokens.extend(tokenizer.tokenize(comments))
        source_tokens.append(tokenizer.sep_token)
        
        # Add code tokens
        code_start = len(source_tokens)
        source_tokens.extend([tokenizer.cls_token] + tokenizer.tokenize(' '.join(code_tokens)))
        
        # Add DFG nodes
        dfg_start = len(source_tokens)
        dfg_to_code = []
        for d in dfg:
            if d[0] not in source_tokens:
                source_tokens.append(d[0])
                dfg_to_code.append((len(source_tokens)-1, d[1]))  # (dfg_pos, code_pos)
        
        # Truncate if too long
        if len(source_tokens) > args.max_source_length - 1:
            source_tokens = source_tokens[:args.max_source_length - 1]
        source_tokens = [tokenizer.cls_token] + source_tokens
        
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * len(source_ids)
        
        # Create position indices (0 for DFG nodes, 2+ for code tokens)
        position_idx = []
        for i in range(len(source_tokens)):
            if i < dfg_start:
                if i >= code_start:
                    position_idx.append(i - code_start + 2)  # Code tokens start at position 2
                else:
                    position_idx.append(1)  # Comments at position 1
            else:
                position_idx.append(0)  # DFG nodes at position 0
        
        # Create attention mask (allowing all tokens to attend to each other)
        attn_mask = [[1] * len(source_tokens) for _ in range(len(source_tokens))]
        
        # Pad to max length
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length
        position_idx += [0] * padding_length
        
        # Pad attention mask
        for i in range(len(attn_mask)):
            attn_mask[i] += [0] * padding_length
        for i in range(padding_length):
            attn_mask.append([0] * args.max_source_length)
        
        # Process target (correct code) for training
        target_ids = None
        target_mask = None
        if stage == "train" and correct_code:
            target_tokens = tokenizer.tokenize(correct_code)
            if len(target_tokens) > args.max_target_length - 2:
                target_tokens = target_tokens[:args.max_target_length - 2]
            target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            target_mask = [1] * len(target_ids)
            
            # Pad target
            target_padding = args.max_target_length - len(target_ids)
            target_ids += [tokenizer.pad_token_id] * target_padding
            target_mask += [0] * target_padding
        
        features.append({
            'source_ids': torch.tensor(source_ids, dtype=torch.long),
            'source_mask': torch.tensor(source_mask, dtype=torch.long),
            'position_idx': torch.tensor(position_idx, dtype=torch.long),
            'attn_mask': torch.tensor(attn_mask, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long) if target_ids else None,
            'target_mask': torch.tensor(target_mask, dtype=torch.long) if target_mask else None,
        })
    
    return features


def create_sample_data():
    """Create sample Verilog error correction data for testing"""
    examples = [
        {
            'buggy_code': 'module test(input a, output b); assign b = a + 1; endmodule',
            'correct_code': 'module test(input a, output b); assign b = a; endmodule',
            'comments': 'Simple wire connection module'
        },
        {
            'buggy_code': 'always @(posedge clk) begin q <= d + 1; end',
            'correct_code': 'always @(posedge clk) begin q <= d; end',
            'comments': 'Register with clock'
        },
        {
            'buggy_code': 'assign out = in1 & in2 | in3',
            'correct_code': 'assign out = (in1 & in2) | in3',
            'comments': 'Logic expression with parentheses'
        }
    ]
    return examples


def main():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_type", default="roberta", type=str,
                      help="Model type: roberta")
    parser.add_argument("--model_name_or_path", default="microsoft/graphcodebert-base", type=str,
                      help="Path to pre-trained model")
    parser.add_argument("--config_name", default="", type=str,
                      help="Pretrained config name or path")
    parser.add_argument("--tokenizer_name", default="microsoft/graphcodebert-base", type=str,
                      help="Pretrained tokenizer name or path")
    parser.add_argument("--cache_dir", default="", type=str,
                      help="Where do you want to store the pre-trained models")
    
    # Data arguments  
    parser.add_argument("--train_filename", default=None, type=str,
                      help="The train filename. Should contain the .json or .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                      help="The dev filename. Should contain the .json or .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                      help="The test filename. Should contain the .json or .jsonl files for this task.")
    parser.add_argument("--max_source_length", default=256, type=int,
                      help="The maximum total source sequence length after tokenization.")
    parser.add_argument("--max_target_length", default=128, type=int,
                      help="The maximum total target sequence length after tokenization.")
    
    # Training arguments
    parser.add_argument("--do_train", action='store_true',
                      help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                      help="Whether to run evaluation.")
    parser.add_argument("--do_test", action='store_true',
                      help="Whether to run testing.")
    parser.add_argument("--output_dir", default="./saved_models", type=str,
                      help="The output directory where the model predictions and checkpoints will be written.")
    
    # Other parameters
    parser.add_argument("--train_batch_size", default=8, type=int,
                      help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                      help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                      help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                      help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                      help="Linear warmup over warmup_steps.")
    parser.add_argument("--seed", type=int, default=42,
                      help="random seed for initialization")
    parser.add_argument("--beam_size", default=10, type=int,
                      help="beam size for beam search")
    
    args = parser.parse_args()
    
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load tokenizer and model config
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                         cache_dir=args.cache_dir if args.cache_dir else None)
    # Load tokenizer from saved model directory for testing, or from pretrained for training
    if args.do_test and not args.do_train:
        # For test-only mode, load tokenizer from the saved model directory
        tokenizer = tokenizer_class.from_pretrained(args.output_dir,
                                                  cache_dir=args.cache_dir if args.cache_dir else None)
        logger.info(f"Loaded tokenizer from saved model: {args.output_dir}")
    else:
        # For training mode, load from pretrained model
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                  cache_dir=args.cache_dir if args.cache_dir else None)
    
    # Add special tokens for Verilog if needed (only during training)
    if args.do_train:
        special_tokens = ['<mask>', '<sep>', '<pad>', '<unk>']
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    if args.do_train:
        # ============================================================
        # 修改部分: 使用新的数据加载函数
        # ============================================================
        if args.train_filename:
            logger.info(f"Loading training data from {args.train_filename}")
            train_examples = load_rtl_dataset(args.train_filename)
        else:
            logger.warning("No train_filename specified, using sample data for testing")
            train_examples = create_sample_data()
        
        logger.info(f"Number of training examples: {len(train_examples)}")
        
        # Convert to features (无需修改)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage="train")
        
        # Create dataset and dataloader (无需修改)
        all_source_ids = torch.stack([f['source_ids'] for f in train_features])
        all_source_mask = torch.stack([f['source_mask'] for f in train_features])
        all_position_idx = torch.stack([f['position_idx'] for f in train_features])
        all_attn_mask = torch.stack([f['attn_mask'] for f in train_features])
        all_target_ids = torch.stack([f['target_ids'] for f in train_features])
        all_target_mask = torch.stack([f['target_mask'] for f in train_features])
        
        train_dataset = TensorDataset(all_source_ids, all_source_mask, all_position_idx, 
                                    all_attn_mask, all_target_ids, all_target_mask)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                    batch_size=args.train_batch_size)
        
        # Load pre-trained encoder
        encoder = model_class.from_pretrained(args.model_name_or_path, config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
        encoder.resize_token_embeddings(len(tokenizer))
        
        # Create decoder (transformer decoder)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # Create our RTL error correction model
        model = RTLErrorCorrectionModel(encoder, decoder, config, args.beam_size, 
                                      args.max_target_length, tokenizer.cls_token_id, tokenizer.sep_token_id)
        model.to(device)
        
        # Setup optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                  num_training_steps=len(train_dataloader) * args.num_train_epochs)
        
        # Training loop (无需修改)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_examples)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Batch size = {args.train_batch_size}")
        
        model.train()
        tr_loss = 0.0
        global_step = 0
        
        for epoch in range(int(args.num_train_epochs)):
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
                source_ids, source_mask, position_idx, attn_mask, target_ids, target_mask = [x.to(device) for x in batch]
                
                model.zero_grad()
                loss = model(source_ids, source_mask, position_idx, attn_mask, target_ids, target_mask)[0]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                tr_loss += loss.item()
                global_step += 1
                
                if global_step % 100 == 0:
                    logger.info(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item():.4f}")
        
        # Save model
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, 'pytorch_model.bin'))
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Model saved to {args.output_dir}")
    
    if args.do_test:
        # ============================================================
        # 测试评估：加载模型并在测试集上预测
        # ============================================================
        if args.test_filename:
            logger.info(f"Loading test data from {args.test_filename}")
            test_examples = load_rtl_dataset(args.test_filename)
        else:
            logger.info("No test_filename specified, using sample data")
            test_examples = create_sample_data()
        
        logger.info(f"***** Running Test Evaluation *****")
        logger.info(f"  Num test examples = {len(test_examples)}")
        
        # Convert to features
        test_features = convert_examples_to_features(test_examples, tokenizer, args, stage='test')
        
        # Create dataset
        all_source_ids = torch.stack([f['source_ids'] for f in test_features])
        all_source_mask = torch.stack([f['source_mask'] for f in test_features])
        all_position_idx = torch.stack([f['position_idx'] for f in test_features])
        all_attn_mask = torch.stack([f['attn_mask'] for f in test_features])
        
        test_data = TensorDataset(all_source_ids, all_source_mask, all_position_idx, all_attn_mask)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
        
        # Initialize model if not already done (for --do_test only mode)
        if 'model' not in locals():
            logger.info("Initializing model for testing...")
            config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
            
            config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
            encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
            
            # Resize token embeddings to match the saved tokenizer
            encoder.resize_token_embeddings(len(tokenizer))
            
            decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
            decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
            model = RTLErrorCorrectionModel(encoder=encoder, decoder=decoder, config=config,
                                           beam_size=args.beam_size, max_length=args.max_target_length,
                                           sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
            model.to(device)
        
            # Load model weights
            logger.info(f"Loading model from {args.output_dir}/pytorch_model.bin")
            checkpoint = torch.load(os.path.join(args.output_dir, 'pytorch_model.bin'), map_location=device)
            model.load_state_dict(checkpoint)  # Now size should match
        
        model.eval()
        
        predictions = []
        logger.info("Generating predictions...")
        
        for batch in tqdm(test_dataloader, desc="Testing"):
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask, position_idx, attn_mask = batch
            
            with torch.no_grad():
                preds = model(source_ids=source_ids, 
                            source_mask=source_mask,
                            position_idx=position_idx, 
                            attn_mask=attn_mask)
                
                # Decode predictions
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    predictions.append(text)
        
        # Evaluate predictions
        logger.info("\n***** Test Results *****")
        correct = 0
        for i, (example, pred) in enumerate(zip(test_examples, predictions)):
            logger.info(f"\n--- Example {i+1} ---")
            logger.info(f"Buggy Code:    {example['buggy_code']}")
            logger.info(f"Expected Fix:  {example['correct_code']}")
            logger.info(f"Predicted Fix: {pred}")
            logger.info(f"Error Type:    {example.get('error_type', 'N/A')}")
            
            # Simple accuracy check (exact match)
            if pred.strip() == example['correct_code'].strip():
                logger.info(f"Status: ✅ CORRECT")
                correct += 1
            else:
                logger.info(f"Status: ❌ INCORRECT")
        
        accuracy = correct / len(test_examples) * 100
        logger.info(f"\n***** Test Summary *****")
        logger.info(f"Total examples: {len(test_examples)}")
        logger.info(f"Correct predictions: {correct}")
        logger.info(f"Accuracy: {accuracy:.2f}%")
        
        # Save predictions
        pred_file = os.path.join(args.output_dir, 'test_predictions.txt')
        with open(pred_file, 'w') as f:
            for i, (example, pred) in enumerate(zip(test_examples, predictions)):
                f.write(f"Example {i+1}:\n")
                f.write(f"Buggy: {example['buggy_code']}\n")
                f.write(f"Expected: {example['correct_code']}\n")
                f.write(f"Predicted: {pred}\n")
                f.write(f"Error Type: {example.get('error_type', 'N/A')}\n")
                f.write("-" * 80 + "\n")
        
        logger.info(f"Predictions saved to {pred_file}")

if __name__ == "__main__":
    main()

