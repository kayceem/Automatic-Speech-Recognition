"""
BART Large Combined News Summarizer with Hyperparameter Tuning
Fine-tuning BART-large on XSum + CNN/DailyMail datasets with systematic hyperparameter optimization
"""

# ============================================================================
# 1. IMPORTS AND SETUP
# ============================================================================

import os
import json
import logging
from datetime import datetime
from itertools import product
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import pipeline
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
from transformers import GenerationConfig
from transformers import DataCollatorForSeq2Seq
from transformers import BartTokenizer, BartForConditionalGeneration

import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import nltk
nltk.download('punkt_tab')

# ============================================================================
# 2. LOGGING SETUP
# ============================================================================

# Create output directories
timestamp = datetime.now().strftime("%Y%m%d")
output_dir = f"bart_combined_output_{timestamp}"
figures_dir = os.path.join(output_dir, "figures")
tokenized_dir = os.path.join("tokenized_data")
metrics_dir = os.path.join(output_dir, "metrics")
logs_dir = os.path.join(output_dir, "logs")
tuning_dir = os.path.join(output_dir, "hyperparameter_tuning")

os.makedirs(figures_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(tuning_dir, exist_ok=True)

# Configure logging
log_file = os.path.join(logs_dir, f"training_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("BART COMBINED NEWS SUMMARIZER WITH HYPERPARAMETER TUNING")
logger.info("="*80)
logger.info(f"Output directory: {output_dir}")
logger.info(f"Tuning directory: {tuning_dir}")

# Configuration
seed = 42
colormap = 'viridis'
template = 'plotly_light'

# Check GPU availability
if torch.cuda.is_available():
    logger.info("GPU is available. Using GPU")
    device = torch.device('cuda')
    logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
else:
    logger.info("GPU is not available. Using CPU")
    device = torch.device('cpu')

# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================

def save_metrics_to_file(metrics_dict, filename, mode='w'):
    """Save metrics to a text file"""
    filepath = os.path.join(metrics_dir, filename)
    with open(filepath, mode) as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        for key, value in metrics_dict.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
    logger.info(f"Metrics saved to {filepath}")

def save_metrics_to_json(metrics_dict, filename):
    """Save metrics to a JSON file"""
    filepath = os.path.join(metrics_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=4, default=str)
    logger.info(f"Metrics saved to JSON: {filepath}")

# ============================================================================
# 4. LOAD AND COMBINE DATASETS
# ============================================================================

from datasets import load_dataset, concatenate_datasets

logger.info("\n" + "="*80)
logger.info("LOADING XSUM AND CNN/DAILYMAIL DATASETS")
logger.info("="*80)

# Load datasets
xsum_dataset = load_dataset("xsum", trust_remote_code=True)
cnn_dataset = load_dataset("cnn_dailymail", "3.0.0", trust_remote_code=True)

# Rename columns
cnn_dataset = cnn_dataset.rename_column("article", "document")
cnn_dataset = cnn_dataset.rename_column("highlights", "summary")

if 'id' in cnn_dataset['train'].column_names:
    cnn_dataset = cnn_dataset.remove_columns(['id'])

# Combine datasets
combined_train = concatenate_datasets([xsum_dataset['train'], cnn_dataset['train']])
combined_test = concatenate_datasets([xsum_dataset['test'], cnn_dataset['test']])
combined_val = concatenate_datasets([xsum_dataset['validation'], cnn_dataset['validation']])

logger.info(f"Combined Train: {len(combined_train):,}")
logger.info(f"Combined Val: {len(combined_val):,}")
logger.info(f"Combined Test: {len(combined_test):,}")


# ============================================================================
# 21. DATASET COMPARISON VISUALIZATION
# ============================================================================
train = combined_train.to_pandas()
test = combined_test.to_pandas()
val = combined_val.to_pandas()

df_text_length = pd.DataFrame()
df_text_length['document'] = train['document'].apply(lambda x: len(str(x).split()))
df_text_length['summary'] = train['summary'].apply(lambda x: len(str(x).split()))
logger.info("\n" + "="*80)
logger.info("CREATING DATASET COMPARISON VISUALIZATION")
logger.info("="*80)

def plot_dataset_comparison():
    """Compare XSum and CNN/DailyMail dataset characteristics"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Dataset Sizes", "Average Text Length"],
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Dataset sizes
    datasets = ['XSum', 'CNN/DM', 'Combined']
    train_sizes = [
        len(xsum_dataset['train']),
        len(cnn_dataset['train']),
        len(combined_train)
    ]
    
    fig.add_trace(
        go.Bar(
            x=datasets,
            y=train_sizes,
            name='Train Size',
            marker_color="#89e09a",
            text=[f'{s:,}' for s in train_sizes],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Average text lengths (calculate from pandas dataframes)
    xsum_doc_avg = xsum_dataset['train'].to_pandas()['document'].apply(lambda x: len(str(x).split())).mean()
    cnn_doc_avg = cnn_dataset['train'].to_pandas()['document'].apply(lambda x: len(str(x).split())).mean()
    # if 'article' in cnn_dataset['train'].columns:
    #     cnn_doc_avg = cnn_dataset['train'].to_pandas()['article'].apply(lambda x: len(str(x).split())).mean()
    # elif 'document' in cnn_dataset['train'].columns:
    # elif 'text' in cnn_dataset['train'].columns:
    #     cnn_doc_avg = cnn_dataset['train'].to_pandas()['text'].apply(lambda x: len(str(x).split())).mean()
    # else:
    #     # Use the first column that's not likely a summary
    #     text_col = [col for col in combined_train.columns if col not in ['summary', 'highlights', 'label']][0]
    #     cnn_doc_avg = combined_train['train'].to_pandas()[text_col].apply(lambda x: len(str(x).split())).mean()

    combined_doc_avg = df_text_length['document'].mean()
    
    fig.add_trace(
        go.Bar(
            x=datasets,
            y=[xsum_doc_avg, cnn_doc_avg, combined_doc_avg],
            name='Avg Document Length',
            marker_color="#95ff00",
            text=[f'{int(v)}' for v in [xsum_doc_avg, cnn_doc_avg, combined_doc_avg]],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='<b>XSum vs CNN/DailyMail Dataset Comparison</b>',
        template=template,
        height=600,
        width=1200,
        showlegend=True
    )
    
    fig.update_yaxes(title_text="<b>Number of Samples</b>", row=1, col=1)
    fig.update_yaxes(title_text="<b>Average Word Count</b>", row=1, col=2)
    
    filepath = os.path.join(figures_dir, 'dataset_comparison.html')
    fig.write_html(filepath)
    logger.info(f"Saved dataset comparison: {filepath}")
    
    try:
        fig.write_image(filepath.replace('.html', '.png'))
    except Exception as e:
        logger.warning(f"Could not save PNG: {e}")
    
    

plot_dataset_comparison()
# ============================================================================
# 5. MODEL AND TOKENIZER SETUP
# ============================================================================

checkpoint = 'facebook/bart-large-xsum'
tokenizer = BartTokenizer.from_pretrained(checkpoint)

# ============================================================================
# 6. DATA PREPROCESSING
# ============================================================================

def preprocess_function(examples):
    """Tokenize documents and summaries"""
    inputs = [doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding=False)
    labels = tokenizer(
        text_target=examples["summary"],
        max_length=128, 
        truncation=True, 
        padding=False
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

from datasets import load_from_disk

# Tokenize datasets (with caching)
tokenized_train_path = os.path.join(tokenized_dir, 'train')
tokenized_val_path = os.path.join(tokenized_dir, 'val')
tokenized_test_path = os.path.join(tokenized_dir, 'test')

if os.path.exists(tokenized_train_path):
    tokenized_train = load_from_disk(tokenized_train_path)
else:
    tokenized_train = combined_train.map(
        preprocess_function, 
        batched=True,
        remove_columns=['document', 'summary']
    )
    tokenized_train.save_to_disk(tokenized_train_path)

if os.path.exists(tokenized_val_path):
    tokenized_val = load_from_disk(tokenized_val_path)
else:
    tokenized_val = combined_val.map(
        preprocess_function,
        batched=True,
        remove_columns=['document', 'summary']
    )
    tokenized_val.save_to_disk(tokenized_val_path)

if os.path.exists(tokenized_test_path):
    tokenized_test = load_from_disk(tokenized_test_path)
else:
    tokenized_test = combined_test.map(
        preprocess_function,
        batched=True,
        remove_columns=['document', 'summary']
    )
    tokenized_test.save_to_disk(tokenized_test_path)

# Use smaller subsets for tuning (adjust as needed based on resources)
TUNING_TRAIN_SIZE = 1000  # Adjust based on available compute
TUNING_VAL_SIZE = 250

def is_within_token_limit(example):
    """Check if document is within BART's 1024 token limit"""
    # Tokenize on-the-fly to check length
    tokens = tokenizer(example['document'], truncation=False, return_tensors=None)
    return len(tokens['input_ids']) <= 1024

tokenized_train_subset = tokenized_train.select(range(min(TUNING_TRAIN_SIZE, len(tokenized_train))))
tokenized_val_subset = tokenized_val.select(range(min(TUNING_VAL_SIZE, len(tokenized_val))))

logger.info(f"Using {len(tokenized_train_subset)} samples for training during tuning")
logger.info(f"Using {len(tokenized_val_subset)} samples for validation during tuning")

# ============================================================================
# 7. EVALUATION METRICS
# ============================================================================

import evaluate
metric = evaluate.load('rouge')

def compute_metrics(eval_pred):
    """Compute ROUGE metrics for summarization"""
    predictions, labels = eval_pred
    
    # Replace -100 in labels (used for padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Clip predictions to valid token range to avoid decoding errors
    vocab_size = len(tokenizer)
    predictions = np.clip(predictions, 0, vocab_size - 1)
    
    # Decode predictions and labels with error handling
    try:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Decoding error: {e}")
        # Fallback: decode one by one and skip problematic ones
        decoded_preds = []
        decoded_labels = []
        for pred, label in zip(predictions, labels):
            try:
                decoded_preds.append(tokenizer.decode(pred, skip_special_tokens=True))
                decoded_labels.append(tokenizer.decode(label, skip_special_tokens=True))
            except:
                decoded_preds.append("")
                decoded_labels.append("")
    
    # Sentence tokenization
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds if pred.strip()]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels if label.strip()]
    
    # Ensure we have matching lengths
    if len(decoded_preds) != len(decoded_labels):
        logger.warning(f"Mismatch in decoded lengths: preds={len(decoded_preds)}, labels={len(decoded_labels)}")
        min_len = min(len(decoded_preds), len(decoded_labels))
        decoded_preds = decoded_preds[:min_len]
        decoded_labels = decoded_labels[:min_len]
    
    # Compute ROUGE metrics
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}
    
    # Calculate generation length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

# ============================================================================
# 8. HYPERPARAMETER TUNING - STAGE 1: TRAINING OPTIMIZATION
# ============================================================================

logger.info("\n" + "="*80)
logger.info("STAGE 1: TRAINING OPTIMIZATION HYPERPARAMETER TUNING")
logger.info("="*80)

# # Define hyperparameter grid for Stage 1
stage1_hyperparams = {
    'learning_rate': [1e-5, 2e-5, 3e-5],
    'num_train_epochs': [3, 4],
    'warmup_ratio': [0, 0.05, 0.1]
}

# stage1_results = []
# best_stage1_config = None
# best_stage1_score = float('-inf')

# # Generate all combinations
# stage1_combinations = list(product(
#     stage1_hyperparams['learning_rate'],
#     stage1_hyperparams['num_train_epochs'],
#     stage1_hyperparams['warmup_ratio']
# ))

# logger.info(f"Total Stage 1 combinations to test: {len(stage1_combinations)}")

# for idx, (lr, epochs, warmup) in enumerate(stage1_combinations, 1):
#     logger.info(f"\n--- Stage 1 Experiment {idx}/{len(stage1_combinations)} ---")
#     logger.info(f"Learning Rate: {lr}, Epochs: {epochs}, Warmup Ratio: {warmup}")
    
#     # Create fresh model instance
#     model = BartForConditionalGeneration.from_pretrained(checkpoint)
#     model.gradient_checkpointing_enable()
    
#     # Create data collator
#     data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
#     # Configure training arguments
#     exp_output_dir = os.path.join(tuning_dir, f"stage1_exp{idx}")
    
#     training_args = Seq2SeqTrainingArguments(
#         output_dir=exp_output_dir,
#         eval_strategy="epoch",
#         save_strategy="no",  # Don't save intermediate checkpoints during tuning
#         learning_rate=lr,
#         per_device_train_batch_size=2,
#         per_device_eval_batch_size=2,
#         gradient_accumulation_steps=4,
#         weight_decay=0.01,
#         num_train_epochs=epochs,
#         warmup_ratio=warmup,
#         predict_with_generate=True,
#         fp16=torch.cuda.is_available(),
#         report_to="none",
#         logging_steps=50,
#         seed=seed,
#         metric_for_best_model='eval_rouge2',
#         greater_is_better=True
#     )
    
#     # Initialize trainer
#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_train_subset,
#         eval_dataset=tokenized_val_subset,
#         processing_class=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics,
#     )
    
#     # Train
#     train_result = trainer.train()
    
#     # Evaluate
#     eval_results = trainer.evaluate()
    
#     # Extract key metrics
#     rouge2_score = eval_results.get('eval_rouge2', 0)
    
#     # Store results
#     result = {
#         'experiment_id': idx,
#         'learning_rate': lr,
#         'num_epochs': epochs,
#         'warmup_ratio': warmup,
#         'train_loss': train_result.metrics.get('train_loss', None),
#         'eval_loss': eval_results.get('eval_loss', None),
#         'eval_rouge1': eval_results.get('eval_rouge1', None),
#         'eval_rouge2': eval_results.get('eval_rouge2', None),
#         'eval_rougeL': eval_results.get('eval_rougeL', None),
#         'eval_rougeLsum': eval_results.get('eval_rougeLsum', None)
#     }
    
#     stage1_results.append(result)
#     logger.info(f"ROUGE-2 Score: {rouge2_score:.4f}")
    
#     # Track best configuration
#     if rouge2_score > best_stage1_score:
#         best_stage1_score = rouge2_score
#         best_stage1_config = {
#             'learning_rate': lr,
#             'num_epochs': epochs,
#             'warmup_ratio': warmup
#         }
#         # Save the best model
#         best_model_dir = os.path.join(tuning_dir, "best_stage1_model")
#         trainer.save_model(best_model_dir)
#         tokenizer.save_pretrained(best_model_dir)
    
#     # Clean up to save memory
#     del model
#     del trainer
#     torch.cuda.empty_cache()

# # Save Stage 1 results
# stage1_df = pd.DataFrame(stage1_results)
# stage1_df.to_csv(os.path.join(tuning_dir, 'stage1_results.csv'), index=False)
# save_metrics_to_json({'results': stage1_results, 'best_config': best_stage1_config}, 
#                      'stage1_tuning_results.json')

# logger.info(f"\n{'='*80}")
# logger.info(f"STAGE 1 BEST CONFIGURATION:")
# logger.info(f"Learning Rate: {best_stage1_config['learning_rate']}")
# logger.info(f"Epochs: {best_stage1_config['num_epochs']}")
# logger.info(f"Warmup Ratio: {best_stage1_config['warmup_ratio']}")
# logger.info(f"Best ROUGE-2 Score: {best_stage1_score:.4f}")
# logger.info(f"{'='*80}")
stage1_df = pd.read_csv(os.path.join(tuning_dir, 'stage1_results.csv'))
with open(os.path.join(metrics_dir, 'stage1_tuning_results.json'), 'r') as f:
    stage1_tuning_data = json.load(f)
best_stage1_config = stage1_tuning_data['best_config']
stage1_results = stage1_tuning_data['results']
best_stage1_score = max([r['eval_rouge2'] for r in stage1_tuning_data['results']])

# ============================================================================
# 9. HYPERPARAMETER TUNING - STAGE 2: GENERATION QUALITY
# ============================================================================

logger.info("\n" + "="*80)
logger.info("STAGE 2: GENERATION QUALITY HYPERPARAMETER TUNING")
logger.info("="*80)

# # Load the best model from Stage 1
# best_model_dir = os.path.join(tuning_dir, "best_stage1_model")
# model = BartForConditionalGeneration.from_pretrained(best_model_dir)

# # Define hyperparameter grid for Stage 2
# stage2_hyperparams = {
#     'num_beams': [2, 4, 6],
#     'length_penalty': [1.5, 2.0, 2.5],
#     'max_length': [96, 128]
# }

# stage2_results = []
# best_stage2_config = None
# best_stage2_score = float('-inf')

# # Generate all combinations
# stage2_combinations = list(product(
#     stage2_hyperparams['num_beams'],
#     stage2_hyperparams['length_penalty'],
#     stage2_hyperparams['max_length']
# ))

# logger.info(f"Total Stage 2 combinations to test: {len(stage2_combinations)}")

# # Create a pipeline for generation
# summarizer = pipeline('summarization', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# # Sample validation set for generation tuning
# val_sample = combined_val.select(range(4000)).filter(is_within_token_limit).select(range(TUNING_VAL_SIZE))  # Use 200 samples for generation tuning

# for idx, (beams, penalty, max_len) in enumerate(stage2_combinations, 1):
#     logger.info(f"\n--- Stage 2 Experiment {idx}/{len(stage2_combinations)} ---")
#     logger.info(f"Beams: {beams}, Length Penalty: {penalty}, Max Length: {max_len}")
    
#     rouge_scores = []
    
#     for sample in val_sample:
#         # Generate summary
#         gen_summary = summarizer(
#             sample['document'],
#             max_length=max_len,
#             min_length=20,
#             num_beams=beams,
#             length_penalty=penalty,
#             no_repeat_ngram_size=3,
#             early_stopping=True,
#             do_sample=False
#         )
        
#         generated = gen_summary[0]['summary_text']
        
#         # Compute ROUGE score for this sample
#         scores = metric.compute(
#             predictions=[generated],
#             references=[sample['summary']],
#             use_stemmer=True
#         )
#         rouge_scores.append(scores)
    
#     # Average ROUGE scores
#     avg_rouge1 = np.mean([s['rouge1'] for s in rouge_scores]) * 100
#     avg_rouge2 = np.mean([s['rouge2'] for s in rouge_scores]) * 100
#     avg_rougeL = np.mean([s['rougeL'] for s in rouge_scores]) * 100
    
#     # Store results
#     result = {
#         'experiment_id': idx,
#         'num_beams': beams,
#         'length_penalty': penalty,
#         'max_length': max_len,
#         'rouge1': avg_rouge1,
#         'rouge2': avg_rouge2,
#         'rougeL': avg_rougeL
#     }
    
#     stage2_results.append(result)
#     logger.info(f"ROUGE-2 Score: {avg_rouge2:.4f}")
    
#     # Track best configuration
#     if avg_rouge2 > best_stage2_score:
#         best_stage2_score = avg_rouge2
#         best_stage2_config = {
#             'num_beams': beams,
#             'length_penalty': penalty,
#             'max_length': max_len
#         }

#     stage2_df = pd.DataFrame(stage2_results)
#     stage2_df.to_csv(os.path.join(tuning_dir, 'stage2_results.csv'), index=False)
#     save_metrics_to_json({'results': stage2_results, 'best_config': best_stage2_config}, 
#                         'stage2_tuning_results.json')
# # Save Stage 2 results
# stage2_df = pd.DataFrame(stage2_results)
# stage2_df.to_csv(os.path.join(tuning_dir, 'stage2_results.csv'), index=False)
# save_metrics_to_json({'results': stage2_results, 'best_config': best_stage2_config}, 
#                      'stage2_tuning_results.json')

# logger.info(f"\n{'='*80}")
# logger.info(f"STAGE 2 BEST CONFIGURATION:")
# logger.info(f"Num Beams: {best_stage2_config['num_beams']}")
# logger.info(f"Length Penalty: {best_stage2_config['length_penalty']}")
# logger.info(f"Max Length: {best_stage2_config['max_length']}")
# logger.info(f"Best ROUGE-2 Score: {best_stage2_score:.4f}")
# logger.info(f"{'='*80}")
stage2_df = pd.read_csv(os.path.join(tuning_dir, 'stage2_results.csv'))
with open(os.path.join(metrics_dir, 'stage2_tuning_results.json'), 'r') as f:
    stage2_tuning_data = json.load(f)
best_stage2_config = stage2_tuning_data['best_config']
stage2_results = stage2_tuning_data['results']
best_stage2_score = max([r['rouge2'] for r in stage2_tuning_data['results']])

# ============================================================================
# 10. VISUALIZATION OF HYPERPARAMETER TUNING RESULTS
# ============================================================================

logger.info("\n" + "="*80)
logger.info("CREATING HYPERPARAMETER TUNING VISUALIZATIONS")
logger.info("="*80)

# Function to create Stage 1 visualizations
def plot_stage1_results(results_df):
    """Create comprehensive visualizations for Stage 1 tuning"""
    
    # 1. Heatmap for Learning Rate vs Epochs
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('ROUGE-2 Scores Heatmap', 'Learning Rate Impact',
                       'Epochs Impact', 'Warmup Ratio Impact'),
        specs=[[{'type': 'heatmap'}, {'type': 'scatter'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Create pivot table for heatmap
    for warmup in stage1_hyperparams['warmup_ratio']:
        warmup_data = results_df[results_df['warmup_ratio'] == warmup]
        pivot = warmup_data.pivot_table(values='eval_rouge2', 
                                        index='num_epochs', 
                                        columns='learning_rate')
        
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=[str(x) for x in pivot.columns],
                y=[str(y) for y in pivot.index],
                colorscale='Viridis',
                text=np.round(pivot.values, 2),
                texttemplate='%{text}',
                name=f'Warmup={warmup}'
            ),
            row=1, col=1
        )
    
    # Learning rate impact
    lr_avg = results_df.groupby('learning_rate')['eval_rouge2'].mean()
    fig.add_trace(
        go.Scatter(
            x=lr_avg.index,
            y=lr_avg.values,
            mode='lines+markers',
            marker=dict(size=10),
            name='Avg ROUGE-2'
        ),
        row=1, col=2
    )
    
    # Epochs impact
    epochs_avg = results_df.groupby('num_epochs')['eval_rouge2'].mean()
    fig.add_trace(
        go.Bar(
            x=epochs_avg.index,
            y=epochs_avg.values,
            marker_color='lightblue',
            name='Avg ROUGE-2'
        ),
        row=2, col=1
    )
    
    # Warmup ratio impact
    warmup_avg = results_df.groupby('warmup_ratio')['eval_rouge2'].mean()
    fig.add_trace(
        go.Bar(
            x=warmup_avg.index,
            y=warmup_avg.values,
            marker_color='lightgreen',
            name='Avg ROUGE-2'
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Learning Rate", row=1, col=1)
    fig.update_yaxes(title_text="Epochs", row=1, col=1)
    fig.update_xaxes(title_text="Learning Rate", row=1, col=2)
    fig.update_yaxes(title_text="ROUGE-2 Score", row=1, col=2)
    fig.update_xaxes(title_text="Epochs", row=2, col=1)
    fig.update_yaxes(title_text="ROUGE-2 Score", row=2, col=1)
    fig.update_xaxes(title_text="Warmup Ratio", row=2, col=2)
    fig.update_yaxes(title_text="ROUGE-2 Score", row=2, col=2)
    
    fig.update_layout(
        title='<b>Stage 1: Training Optimization Results</b>',
        template=template,
        height=800,
        showlegend=False
    )
    
    filepath = os.path.join(figures_dir, 'stage1_tuning_results.html')
    fig.write_html(filepath)
    logger.info(f"Saved Stage 1 visualization: {filepath}")

# Function to create Stage 2 visualizations
def plot_stage2_results(results_df):
    """Create comprehensive visualizations for Stage 2 tuning"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('3D Surface Plot', 'Beam Width Impact',
                       'Length Penalty Impact', 'Max Length Impact'),
        specs=[[{'type': 'surface'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # 3D surface plot
    pivot = results_df.pivot_table(values='rouge2', 
                                   index='num_beams', 
                                   columns='length_penalty',
                                   aggfunc='mean')
    
    fig.add_trace(
        go.Surface(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Viridis',
            name='ROUGE-2'
        ),
        row=1, col=1
    )
    
    # Beam width impact
    beams_avg = results_df.groupby('num_beams')[['rouge1', 'rouge2', 'rougeL']].mean()
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        fig.add_trace(
            go.Scatter(
                x=beams_avg.index,
                y=beams_avg[metric],
                mode='lines+markers',
                name=metric.upper()
            ),
            row=1, col=2
        )
    
    # Length penalty impact
    penalty_avg = results_df.groupby('length_penalty')['rouge2'].mean()
    fig.add_trace(
        go.Scatter(
            x=penalty_avg.index,
            y=penalty_avg.values,
            mode='lines+markers',
            marker=dict(size=10, color='orange'),
            name='Avg ROUGE-2'
        ),
        row=2, col=1
    )
    
    # Max length impact
    maxlen_avg = results_df.groupby('max_length')['rouge2'].mean()
    fig.add_trace(
        go.Scatter(
            x=maxlen_avg.index,
            y=maxlen_avg.values,
            mode='lines+markers',
            marker=dict(size=10, color='green'),
            name='Avg ROUGE-2'
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Length Penalty", row=1, col=1)
    fig.update_yaxes(title_text="Beam Width", row=1, col=1)
    fig.update_xaxes(title_text="Beam Width", row=1, col=2)
    fig.update_yaxes(title_text="ROUGE Score", row=1, col=2)
    fig.update_xaxes(title_text="Length Penalty", row=2, col=1)
    fig.update_yaxes(title_text="ROUGE-2 Score", row=2, col=1)
    fig.update_xaxes(title_text="Max Length", row=2, col=2)
    fig.update_yaxes(title_text="ROUGE-2 Score", row=2, col=2)
    
    fig.update_layout(
        title='<b>Stage 2: Generation Quality Results</b>',
        template=template,
        height=800,
        showlegend=True
    )
    
    filepath = os.path.join(figures_dir, 'stage2_tuning_results.html')
    fig.write_html(filepath)
    logger.info(f"Saved Stage 2 visualization: {filepath}")

# Create comparison plot
def plot_tuning_comparison():
    """Create comparison plot of all hyperparameter combinations"""
    
    fig = go.Figure()
    
    # Stage 1 results
    stage1_scores = [r['eval_rouge2'] for r in stage1_results]
    stage1_labels = [f"LR:{r['learning_rate']:.0e}<br>E:{r['num_epochs']}<br>W:{r['warmup_ratio']}" 
                     for r in stage1_results]
    
    fig.add_trace(go.Bar(
        x=list(range(len(stage1_scores))),
        y=stage1_scores,
        text=stage1_labels,
        hovertemplate='%{text}<br>ROUGE-2: %{y:.2f}<extra></extra>',
        name='Stage 1 Experiments',
        marker_color='lightblue'
    ))
    
    # Mark best Stage 1 configuration
    best_idx = stage1_scores.index(max(stage1_scores))
    fig.add_annotation(
        x=best_idx,
        y=stage1_scores[best_idx],
        text="Best Stage 1",
        showarrow=True,
        arrowhead=2,
        arrowcolor="red"
    )
    
    fig.update_layout(
        title='<b>All Hyperparameter Tuning Experiments</b>',
        xaxis_title='<b>Experiment ID</b>',
        yaxis_title='<b>ROUGE-2 Score</b>',
        template=template,
        height=600,
        width=1200
    )
    
    filepath = os.path.join(figures_dir, 'all_tuning_experiments.html')
    fig.write_html(filepath)
    logger.info(f"Saved comparison visualization: {filepath}")

# # Generate all visualizations
# plot_stage1_results(stage1_df)
# plot_stage2_results(stage2_df)
# plot_tuning_comparison()

# ============================================================================
# 11. FINAL TRAINING WITH BEST HYPERPARAMETERS
# ============================================================================
# logger.info("\n" + "="*80)
# logger.info("FINAL TRAINING WITH OPTIMAL HYPERPARAMETERS")
# logger.info("="*80)

# # Combine best configurations
# final_config = {
#     **best_stage1_config,
#     **best_stage2_config
# }

# logger.info("Optimal Configuration:")
# for key, value in final_config.items():
#     logger.info(f"  {key}: {value}")

# save_metrics_to_json(final_config, 'optimal_hyperparameters.json')

# # Train final model with best hyperparameters on full dataset
# logger.info("\nTraining final model with optimal hyperparameters...")

# # Load fresh model
# final_model = BartForConditionalGeneration.from_pretrained(checkpoint)
# final_model.gradient_checkpointing_enable()
# # Create data collator
# data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=final_model)

# # Configure training with optimal hyperparameters
# final_output_dir = os.path.join(output_dir, 'final_model')

# generation_config = GenerationConfig(
#     max_length=best_stage2_config['max_length'],
#     min_length=20,
#     num_beams=best_stage2_config['num_beams'],
#     length_penalty=best_stage2_config['length_penalty'],
#     no_repeat_ngram_size=3,
#     early_stopping=True,
# )

# final_model.generation_config = generation_config

# final_training_args = Seq2SeqTrainingArguments(
#     output_dir=final_output_dir,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model='eval_loss',
#     learning_rate=best_stage1_config['learning_rate'],
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     gradient_accumulation_steps=4,
#     weight_decay=0.01,
#     save_total_limit=2,
#     num_train_epochs=best_stage1_config['num_epochs'],
#     warmup_ratio=best_stage1_config['warmup_ratio'],
#     predict_with_generate=True,
#     fp16=torch.cuda.is_available(),
#     report_to="none",
#     logging_steps=100,
#     logging_dir=os.path.join(final_output_dir, 'logs'),  # Added for tensorboard logs
#     seed=seed,
#     generation_config=generation_config
# )

# # Custom callback to track training/validation loss history
# class LossHistoryCallback(TrainerCallback):
#     def __init__(self):
#         self.train_loss_history = []
#         self.eval_loss_history = []
#         self.eval_metrics_history = []
        
#     def on_log(self, args, state, control, logs=None, **kwargs):
#         if logs is not None:
#             if 'loss' in logs:
#                 self.train_loss_history.append({
#                     'epoch': logs.get('epoch', state.epoch),
#                     'step': state.global_step,
#                     'loss': logs['loss']
#                 })
    
#     def on_evaluate(self, args, state, control, metrics=None, **kwargs):
#         if metrics is not None:
#             eval_entry = {
#                 'epoch': state.epoch,
#                 'step': state.global_step,
#             }
#             eval_entry.update(metrics)
#             self.eval_metrics_history.append(eval_entry)
            
#             if 'eval_loss' in metrics:
#                 self.eval_loss_history.append({
#                     'epoch': state.epoch,
#                     'step': state.global_step,
#                     'eval_loss': metrics['eval_loss']
#                 })

# # Initialize callback
# loss_callback = LossHistoryCallback()

# # Initialize final trainer
# final_trainer = Seq2SeqTrainer(
#     model=final_model,
#     args=final_training_args,
#     train_dataset=tokenized_train.select(range(min(TUNING_TRAIN_SIZE*3, len(tokenized_train)))),  # Use full dataset
#     eval_dataset=tokenized_val.select(range(min(TUNING_VAL_SIZE*3, len(tokenized_val)))),     # Use full validation set
#     processing_class=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     callbacks=[loss_callback]  # Add custom callback
# )

# # Train final model
# logger.info("Starting final training...")
# final_train_result = final_trainer.train()

# # Final evaluation
# logger.info("Evaluating on validation set...")
# final_val_results = final_trainer.evaluate()

# logger.info("Evaluating on test set...")
# final_test_results = final_trainer.evaluate(
#     eval_dataset=tokenized_test.select(range(min(TUNING_VAL_SIZE*3, len(tokenized_test))))
# )

# # Save comprehensive results with loss history
# final_results = {
#     'training_metrics': final_train_result.metrics,
#     'validation_metrics': final_val_results,
#     'test_metrics': final_test_results,
#     'optimal_hyperparameters': final_config,
#     'train_loss_history': loss_callback.train_loss_history,
#     'eval_loss_history': loss_callback.eval_loss_history,
#     'eval_metrics_history': loss_callback.eval_metrics_history
# }

# save_metrics_to_json(final_results, 'final_model_results.json')

# # Also save loss history separately for easy plotting
# save_metrics_to_json({
#     'train_loss': loss_callback.train_loss_history,
#     'eval_loss': loss_callback.eval_loss_history,
#     'eval_metrics': loss_callback.eval_metrics_history
# }, 'final_training_history.json')

# # Create simple loss curve plot
# try:
#     import matplotlib.pyplot as plt
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
#     # Training loss
#     if loss_callback.train_loss_history:
#         train_steps = [x['step'] for x in loss_callback.train_loss_history]
#         train_losses = [x['loss'] for x in loss_callback.train_loss_history]
#         ax1.plot(train_steps, train_losses, label='Training Loss', color='blue', alpha=0.7)
#         ax1.set_xlabel('Steps')
#         ax1.set_ylabel('Loss')
#         ax1.set_title('Training Loss Over Time')
#         ax1.legend()
#         ax1.grid(True, alpha=0.3)
    
#     # Validation loss
#     if loss_callback.eval_loss_history:
#         eval_steps = [x['step'] for x in loss_callback.eval_loss_history]
#         eval_losses = [x['eval_loss'] for x in loss_callback.eval_loss_history]
#         ax2.plot(eval_steps, eval_losses, label='Validation Loss', color='red', marker='o')
#         ax2.set_xlabel('Steps')
#         ax2.set_ylabel('Loss')
#         ax2.set_title('Validation Loss Over Time')
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'final_training_loss_curves.png'), dpi=300, bbox_inches='tight')
#     plt.close()
#     logger.info(f"Loss curves saved to: {os.path.join(output_dir, 'final_training_loss_curves.png')}")
# except Exception as e:
#     logger.warning(f"Could not generate loss curve plot: {e}")

# # Log final metrics summary
# logger.info("\n" + "="*80)
# logger.info("FINAL TRAINING SUMMARY")
# logger.info("="*80)
# logger.info(f"Final Training Loss: {final_train_result.metrics.get('train_loss', 'N/A'):.4f}")
# logger.info(f"Final Validation Loss: {final_val_results.get('eval_loss', 'N/A'):.4f}")
# logger.info(f"Final Test Loss: {final_test_results.get('eval_loss', 'N/A'):.4f}")
# logger.info(f"Validation ROUGE-2: {final_val_results.get('eval_rouge2', 'N/A'):.4f}")
# logger.info(f"Test ROUGE-2: {final_test_results.get('eval_rouge2', 'N/A'):.4f}")

# # Save final model
# logger.info("\nSaving final model...")
# final_trainer.save_model(os.path.join(output_dir, "bart_finetuned_optimized"))
# tokenizer.save_pretrained(os.path.join(output_dir, "bart_finetuned_optimized"))

# logger.info("\n" + "="*80)
# logger.info("HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY")
# logger.info("="*80)
# logger.info(f"All outputs saved to: {output_dir}")
# logger.info(f"Optimal hyperparameters saved to: optimal_hyperparameters.json")
# logger.info(f"Training history saved to: final_training_history.json")
# logger.info(f"Final model saved to: bart_finetuned_optimized")
# logger.info("="*80)

# # Create summary report
# summary_report = f"""
# HYPERPARAMETER TUNING SUMMARY REPORT
# =====================================

# STAGE 1 - TRAINING OPTIMIZATION
# Best Configuration:
# - Learning Rate: {best_stage1_config['learning_rate']}
# - Epochs: {best_stage1_config['num_epochs']}
# - Warmup Ratio: {best_stage1_config['warmup_ratio']}
# - Best ROUGE-2: {best_stage1_score:.4f}

# STAGE 2 - GENERATION QUALITY
# Best Configuration:
# - Num Beams: {best_stage2_config['num_beams']}
# - Length Penalty: {best_stage2_config['length_penalty']}
# - Max Length: {best_stage2_config['max_length']}
# - Best ROUGE-2: {best_stage2_score:.4f}

# FINAL MODEL PERFORMANCE
# - Validation ROUGE-1: {final_val_results.get('eval_rouge1', 'N/A'):.2f}
# - Validation ROUGE-2: {final_val_results.get('eval_rouge2', 'N/A'):.2f}
# - Validation ROUGE-L: {final_val_results.get('eval_rougeL', 'N/A'):.2f}
# - Test ROUGE-1: {final_test_results.get('eval_rouge1', 'N/A'):.2f}
# - Test ROUGE-2: {final_test_results.get('eval_rouge2', 'N/A'):.2f}
# - Test ROUGE-L: {final_test_results.get('eval_rougeL', 'N/A'):.2f}
# """

# with open(os.path.join(output_dir, 'tuning_summary_report.txt'), 'w') as f:
#     f.write(summary_report)

# logger.info(summary_report)