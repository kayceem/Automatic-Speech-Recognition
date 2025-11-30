"""
BART Large Combined News Summarizer
Fine-tuning BART-large on XSum + CNN/DailyMail datasets for news article summarization
With proper logging, figure saving, and metrics tracking
"""

# ============================================================================
# 1. IMPORTS AND SETUP
# ============================================================================

import os
import json
import logging
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import pipeline
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
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

os.makedirs(figures_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

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
logger.info("BART COMBINED NEWS SUMMARIZER (XSum + CNN/DailyMail)")
logger.info("="*80)
logger.info(f"Output directory: {output_dir}")
logger.info(f"Figures directory: {figures_dir}")
logger.info(f"Metrics directory: {metrics_dir}")
logger.info(f"Log file: {log_file}")

# Configuration
seed = 42
colormap = 'cividis'
template = 'plotly_dark'

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

# def display_feature_list(features, feature_type):
#     """Display features within each list for each type of data"""
#     logger.info(f"\n{feature_type} Features: ")
#     logger.info(', '.join(features) if features else 'None')

# def describe_df(df, name="Dataset"):
#     """Print basic info on the dataset and set global variables for feature lists"""
#     global categorical_features, continuous_features, binary_features
#     categorical_features = [col for col in df.columns if df[col].dtype == 'object']
#     binary_features = [col for col in df.columns if df[col].nunique() <= 2 and df[col].dtype != 'object']
#     continuous_features = [col for col in df.columns if df[col].dtype != 'object' and col not in binary_features]

#     info_dict = {
#         "shape": df.shape,
#         "samples": df.shape[0],
#         "attributes": df.shape[1],
#         "missing_data": df.isnull().sum().to_dict(),
#         "duplicates": int(df.duplicated().sum()),
#         "dtypes": df.dtypes.to_dict()
#     }

#     logger.info(f"\n{name} shape: {df.shape}")
#     logger.info(f"{df.shape[0]:,.0f} samples")
#     logger.info(f"{df.shape[1]:,.0f} attributes")
#     logger.info(f'Missing Data: \n{df.isnull().sum()}')
#     logger.info(f'Duplicates: {df.duplicated().sum()}')
#     logger.info(f'Data Types: \n{df.dtypes}')

#     display_feature_list(categorical_features, 'Categorical')
#     display_feature_list(continuous_features, 'Continuous')
#     display_feature_list(binary_features, 'Binary')

#     logger.info(f'\n{name} Head: \n{df.head(5)}')
#     logger.info(f'\n{name} Tail: \n{df.tail(5)}')

#     save_metrics_to_json(info_dict, f"{name.lower().replace(' ', '_')}_info.json")

# def histogram_boxplot(df, hist_color, box_color, height, width, legend, name):
#     """Plot a Histogram and a Box Plot side by side"""
#     features = df.select_dtypes(include=[np.number]).columns.tolist()

#     for feat in features:
#         try:
#             fig = make_subplots(
#                 rows=1,
#                 cols=2,
#                 subplot_titles=["Box Plot", "Histogram"],
#                 horizontal_spacing=0.2
#             )

#             density = gaussian_kde(df[feat].dropna())
#             x_vals = np.linspace(df[feat].min(), df[feat].max(), 200)
#             density_vals = density(x_vals)

#             fig.add_trace(go.Scatter(x=x_vals, y=density_vals, mode='lines',
#                                      fill='tozeroy', name="Density", line_color=hist_color), row=1, col=2)
#             fig.add_trace(go.Box(y=df[feat], name="Box Plot", boxmean=True, line_color=box_color), row=1, col=1)

#             fig.update_layout(
#                 title={'text': f'<b>{name} Word Count<br><sup><i>&nbsp;&nbsp;&nbsp;&nbsp;{feat}</i></sup></b>',
#                        'x': .025, 'xanchor': 'left'},
#                 margin=dict(t=100),
#                 showlegend=legend,
#                 template=template,
#                 height=height, width=width
#             )

#             fig.update_yaxes(title_text=f"<b>Words</b>", row=1, col=1, showgrid=False)
#             fig.update_xaxes(title_text="", row=1, col=1, showgrid=False)
#             fig.update_yaxes(title_text="<b>Frequency</b>", row=1, col=2, showgrid=False)
#             fig.update_xaxes(title_text=f"<b>Words</b>", row=1, col=2, showgrid=False)

#             filename = f"{name.lower().replace(' ', '_')}_{feat}_distribution.html"
#             filepath = os.path.join(figures_dir, filename)
#             fig.write_html(filepath)
#             logger.info(f"Saved figure: {filepath}")
            
#             try:
#                 png_filepath = filepath.replace('.html', '.png')
#                 fig.write_image(png_filepath)
#                 logger.info(f"Saved PNG: {png_filepath}")
#             except Exception as e:
#                 logger.warning(f"Could not save PNG (kaleido may not be installed): {e}")

            
#         except Exception as e:
#             logger.error(f"An error occurred while plotting {feat}: {e}")

# def plot_correlation(df, title, subtitle, height, width, font_size):
#     """Plot a correlation map among features in the dataset"""
#     corr = np.round(df.corr(numeric_only=True), 2)
#     mask = np.triu(np.ones_like(corr, dtype=bool))
#     c_mask = np.where(~mask, corr, 100)

#     c = []
#     for i in c_mask.tolist()[1:]:
#         c.append([x for x in i if x != 100])

#     fig = ff.create_annotated_heatmap(
#         z=c[::-1],
#         x=corr.index.tolist()[:-1],
#         y=corr.columns.tolist()[1:][::-1],
#         colorscale=colormap
#     )

#     fig.update_layout(
#         title={'text': f"<b>{title} Heatmap<br><sup>&nbsp;&nbsp;&nbsp;&nbsp;<i>{subtitle}</i></sup></b>",
#                'x': .025, 'xanchor': 'left', 'y': .95},
#         margin=dict(t=210, l=110),
#         yaxis=dict(autorange='reversed', showgrid=False),
#         xaxis=dict(showgrid=False),
#         template=template,
#         height=height, width=width
#     )

#     fig.add_trace(go.Heatmap(z=c[::-1], colorscale=colormap, showscale=True, visible=False))
#     fig.data[1].visible = True

#     for i in range(len(fig.layout.annotations)):
#         fig.layout.annotations[i].font.size = font_size

#     filename = f"{title.lower().replace(' ', '_')}_{subtitle.lower().replace(' ', '_')}_heatmap.html"
#     filepath = os.path.join(figures_dir, filename)
#     fig.write_html(filepath)
#     logger.info(f"Saved figure: {filepath}")
    
#     try:
#         png_filepath = filepath.replace('.html', '.png')
#         fig.write_image(png_filepath)
#         logger.info(f"Saved PNG: {png_filepath}")
#     except Exception as e:
#         logger.warning(f"Could not save PNG: {e}")

    

# def compute_tfidf(df_column, ngram_range=(1, 1), max_features=15):
#     """Compute TF-IDF features"""
#     vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=ngram_range)
#     x = vectorizer.fit_transform(df_column.fillna(''))
#     df_tfidfvect = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())
#     return df_tfidfvect

def plot_training_history(history_file, output_dir):
    """Plot training history from trainer state"""
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        log_history = history.get('log_history', [])
        
        # Separate tracking for training and evaluation metrics
        train_data = {'steps': [], 'losses': [], 'epochs': []}
        eval_data = {'steps': [], 'losses': [], 'epochs': []}
        
        for entry in log_history:
            step = entry.get('step', 0)
            epoch = entry.get('epoch', 0)
            
            # Training loss entries (these appear frequently)
            if 'loss' in entry and 'eval_loss' not in entry:
                train_data['steps'].append(step)
                train_data['losses'].append(entry['loss'])
                train_data['epochs'].append(epoch)
            
            # Evaluation loss entries (these appear at epoch boundaries)
            if 'eval_loss' in entry:
                eval_data['steps'].append(step)
                eval_data['losses'].append(entry['eval_loss'])
                eval_data['epochs'].append(epoch)
        
        # Create the plot
        fig = go.Figure()
        
        # Plot training loss
        if train_data['losses']:
            fig.add_trace(go.Scatter(
                x=train_data['steps'],  # Use steps for x-axis
                y=train_data['losses'],
                mode='lines',  # Just lines for many points
                name='Training Loss',
                line=dict(color='#89c2e0', width=2),
                hovertemplate='<b>Training Loss</b><br>' +
                              'Step: %{x}<br>' +
                              'Loss: %{y:.4f}<br>' +
                              'Epoch: %{customdata:.2f}<extra></extra>',
                customdata=train_data['epochs']
            ))
        
        # Plot validation loss
        if eval_data['losses']:
            fig.add_trace(go.Scatter(
                x=eval_data['steps'],  # Use steps for x-axis
                y=eval_data['losses'],
                mode='lines+markers',  # Lines and markers for fewer points
                name='Validation Loss',
                line=dict(color='#d500ff', width=2),
                marker=dict(size=8),
                hovertemplate='<b>Validation Loss</b><br>' +
                              'Step: %{x}<br>' +
                              'Loss: %{y:.4f}<br>' +
                              'Epoch: %{customdata:.1f}<extra></extra>',
                customdata=eval_data['epochs']
            ))
        
        # Add epoch boundaries as vertical lines (optional but helpful)
        if eval_data['steps']:
            for i, step in enumerate(eval_data['steps']):
                fig.add_vline(
                    x=step,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.3,
                    annotation_text=f"Epoch {int(eval_data['epochs'][i])}",
                    annotation_position="top right"
                )
        
        # Update layout
        fig.update_layout(
            title='<b>Training History</b>',
            xaxis_title='<b>Training Steps</b>',
            yaxis_title='<b>Loss</b>',
            template=template,
            height=600,
            width=1000,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Save files
        filepath = os.path.join(output_dir, 'training_history.html')
        fig.write_html(filepath)
        logger.info(f"Saved training history plot: {filepath}")
        
        try:
            fig.write_image(filepath.replace('.html', '.png'))
            logger.info(f"Saved training history image: {filepath.replace('.html', '.png')}")
        except Exception as e:
            logger.warning(f"Could not save PNG: {e}")
        
        # Print summary statistics
        if train_data['losses']:
            logger.info(f"Final training loss: {train_data['losses'][-1]:.4f}")
            logger.info(f"Min training loss: {min(train_data['losses']):.4f}")
        
        if eval_data['losses']:
            logger.info(f"Final validation loss: {eval_data['losses'][-1]:.4f}")
            logger.info(f"Min validation loss: {min(eval_data['losses']):.4f}")
            best_epoch_idx = eval_data['losses'].index(min(eval_data['losses']))
            logger.info(f"Best validation loss at epoch: {eval_data['epochs'][best_epoch_idx]:.1f}")
        
    except Exception as e:
        logger.error(f"Could not plot training history: {e}")
        import traceback
        logger.error(traceback.format_exc())

# ============================================================================
# 4. LOAD AND COMBINE DATASETS
# ============================================================================

from datasets import load_dataset, concatenate_datasets

logger.info("\n" + "="*80)
logger.info("LOADING XSUM AND CNN/DAILYMAIL DATASETS")
logger.info("="*80)

# Load XSum dataset
logger.info("Loading XSum dataset...")
xsum_dataset = load_dataset("xsum", trust_remote_code=True)

# Load CNN/DailyMail dataset
logger.info("Loading CNN/DailyMail dataset...")
cnn_dataset = load_dataset("cnn_dailymail", "3.0.0", trust_remote_code=True)

# Rename columns in CNN/DailyMail to match XSum format
logger.info("Renaming CNN/DailyMail columns to match XSum format...")
cnn_dataset = cnn_dataset.rename_column("article", "document")
cnn_dataset = cnn_dataset.rename_column("highlights", "summary")

# Remove 'id' column from CNN/DailyMail if it exists (to avoid conflicts)
if 'id' in cnn_dataset['train'].column_names:
    cnn_dataset = cnn_dataset.remove_columns(['id'])

# Combine the datasets
logger.info("Combining XSum and CNN/DailyMail datasets...")
combined_train = concatenate_datasets([xsum_dataset['train'], cnn_dataset['train']])
combined_test = concatenate_datasets([xsum_dataset['test'], cnn_dataset['test']])
combined_val = concatenate_datasets([xsum_dataset['validation'], cnn_dataset['validation']])

# Convert to pandas for EDA
train = combined_train.to_pandas()
test = combined_test.to_pandas()
val = combined_val.to_pandas()

dataset_info = {
    "xsum_train_size": len(xsum_dataset['train']),
    "xsum_test_size": len(xsum_dataset['test']),
    "xsum_validation_size": len(xsum_dataset['validation']),
    "cnn_train_size": len(cnn_dataset['train']),
    "cnn_test_size": len(cnn_dataset['test']),
    "cnn_validation_size": len(cnn_dataset['validation']),
    "combined_train_size": len(train),
    "combined_test_size": len(test),
    "combined_validation_size": len(val)
}

logger.info(f"\nXSum Dataset Sizes:")
logger.info(f"  Train: {len(xsum_dataset['train']):,}")
logger.info(f"  Test: {len(xsum_dataset['test']):,}")
logger.info(f"  Validation: {len(xsum_dataset['validation']):,}")

logger.info(f"\nCNN/DailyMail Dataset Sizes:")
logger.info(f"  Train: {len(cnn_dataset['train']):,}")
logger.info(f"  Test: {len(cnn_dataset['test']):,}")
logger.info(f"  Validation: {len(cnn_dataset['validation']):,}")

logger.info(f"\nCombined Dataset Sizes:")
logger.info(f"  Train: {len(train):,}")
logger.info(f"  Test: {len(test):,}")
logger.info(f"  Validation: {len(val):,}")

save_metrics_to_json(dataset_info, "combined_dataset_sizes.json")

# # ============================================================================
# # 5. EXPLORATORY DATA ANALYSIS
# # ============================================================================

# logger.info("\n" + "="*80)
# logger.info("COMBINED TRAINING SET ANALYSIS")
# logger.info("="*80)
# describe_df(train, "Combined Training Set")

# # Analyze text lengths
# logger.info("\n" + "="*80)
# logger.info("TEXT LENGTH ANALYSIS")
# logger.info("="*80)

# df_text_length = pd.DataFrame()
# df_text_length['document'] = train['document'].apply(lambda x: len(str(x).split()))
# df_text_length['summary'] = train['summary'].apply(lambda x: len(str(x).split()))

# doc_stats = df_text_length['document'].describe().to_dict()
# sum_stats = df_text_length['summary'].describe().to_dict()

# logger.info(f"\nDocument statistics:\n{df_text_length['document'].describe()}")
# logger.info(f"\nSummary statistics:\n{df_text_length['summary'].describe()}")

# text_length_stats = {
#     "document_statistics": doc_stats,
#     "summary_statistics": sum_stats
# }
# save_metrics_to_json(text_length_stats, "combined_text_length_statistics.json")

# # Plot distributions
# histogram_boxplot(df_text_length, '#89c2e0', '#d500ff', 600, 1000, True, 'Combined Train Dataset')

# # ============================================================================
# # 6. N-GRAM ANALYSIS
# # ============================================================================

# logger.info("\n" + "="*80)
# logger.info("N-GRAM ANALYSIS - DOCUMENTS")
# logger.info("="*80)

# # Unigrams - Documents
# logger.info("Computing Unigrams for documents...")
# df_tfidf_uni = compute_tfidf(train['document'], ngram_range=(1, 1), max_features=15)
# plot_correlation(df_tfidf_uni, 'Unigrams', 'Combined Train - Document', 800, 800, 12)

# # Bigrams - Documents
# logger.info("Computing Bigrams for documents...")
# df_tfidf_bi = compute_tfidf(train['document'], ngram_range=(2, 2), max_features=15)
# plot_correlation(df_tfidf_bi, 'Bigrams', 'Combined Train - Document', 800, 800, 12)

# # Trigrams - Documents
# logger.info("Computing Trigrams for documents...")
# df_tfidf_tri = compute_tfidf(train['document'], ngram_range=(3, 3), max_features=15)
# plot_correlation(df_tfidf_tri, 'Trigrams', 'Combined Train - Document', 800, 800, 12)

# logger.info("\n" + "="*80)
# logger.info("N-GRAM ANALYSIS - SUMMARIES")
# logger.info("="*80)

# # Bigrams - Summaries
# logger.info("Computing Bigrams for summaries...")
# df_tfidf_bi_sum = compute_tfidf(train['summary'], ngram_range=(2, 2), max_features=15)
# plot_correlation(df_tfidf_bi_sum, 'Bigrams', 'Combined Train - Summary', 800, 800, 12)

# # Trigrams - Summaries
# logger.info("Computing Trigrams for summaries...")
# df_tfidf_tri_sum = compute_tfidf(train['summary'], ngram_range=(3, 3), max_features=15)
# plot_correlation(df_tfidf_tri_sum, 'Trigrams', 'Combined Train - Summary', 800, 800, 12)

# # ============================================================================
# # 7. SAMPLE DATA INSPECTION
# # ============================================================================

# logger.info("\n" + "="*80)
# logger.info("SAMPLE ARTICLES FROM COMBINED DATASET")
# logger.info("="*80)

# samples_info = []
# for i in range(3):
#     sample = {
#         "sample_id": i+1,
#         "document_preview": train['document'].iloc[i][:500],
#         "summary": train['summary'].iloc[i]
#     }
#     samples_info.append(sample)
    
#     logger.info(f"\n--- Sample {i+1} ---")
#     logger.info(f"Document:\n{train['document'].iloc[i][:500]}...")
#     logger.info(f"\nSummary:\n{train['summary'].iloc[i]}")
#     logger.info("-" * 80)

# save_metrics_to_json({"samples": samples_info}, "combined_sample_articles.json")

# ============================================================================
# 8. MODEL SETUP
# ============================================================================

logger.info("\n" + "="*80)
logger.info("LOADING MODEL AND TOKENIZER")
logger.info("="*80)

checkpoint = 'facebook/bart-large-xsum'
logger.info(f"Loading tokenizer from {checkpoint}...")
tokenizer = BartTokenizer.from_pretrained(checkpoint)

logger.info(f"Loading model from {checkpoint}...")
model = BartForConditionalGeneration.from_pretrained(checkpoint)
model.gradient_checkpointing_enable()

logger.info("\nModel architecture:")
logger.info(str(model))

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

model_info = {
    "checkpoint": checkpoint,
    "total_parameters": total_params,
    "trainable_parameters": trainable_params
}
logger.info(f"Total parameters: {total_params:,}")
logger.info(f"Trainable parameters: {trainable_params:,}")

save_metrics_to_json(model_info, "model_info.json")

# ============================================================================
# 9. DATA PREPROCESSING
# ============================================================================

logger.info("\n" + "="*80)
logger.info("PREPROCESSING COMBINED DATA")
logger.info("="*80)

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
# Define paths
tokenized_train_path = os.path.join(tokenized_dir, 'train')
tokenized_test_path = os.path.join(tokenized_dir, 'test')
tokenized_val_path = os.path.join(tokenized_dir, 'val')

# TRAIN
if os.path.exists(tokenized_train_path):
    logger.info("Loading cached tokenized training set...")
    tokenized_train = load_from_disk(tokenized_train_path)
else:
    logger.info("Tokenizing combined training set...")
    tokenized_train = combined_train.map(
        preprocess_function, 
        batched=True,
        batch_size=1000,
        num_proc=os.cpu_count(),
        remove_columns=['document', 'summary'],
        load_from_cache_file=True
    )
    tokenized_train.save_to_disk(tokenized_train_path)
    logger.info(f"Saved tokenized training set to {tokenized_train_path}")

# TEST
if os.path.exists(tokenized_test_path):
    logger.info("Loading cached tokenized test set...")
    tokenized_test = load_from_disk(tokenized_test_path)
else:
    logger.info("Tokenizing combined test set...")
    tokenized_test = combined_test.map(
        preprocess_function,
        batched=True,
        batch_size=1000,
        num_proc=os.cpu_count(),
        remove_columns=['document', 'summary'],
        load_from_cache_file=True
    )
    tokenized_test.save_to_disk(tokenized_test_path)
    logger.info(f"Saved tokenized test set to {tokenized_test_path}")

# VALIDATION
if os.path.exists(tokenized_val_path):
    logger.info("Loading cached tokenized validation set...")
    tokenized_val = load_from_disk(tokenized_val_path)
else:
    logger.info("Tokenizing combined validation set...")
    tokenized_val = combined_val.map(
        preprocess_function,
        batched=True,
        batch_size=1000,
        num_proc=os.cpu_count(),
        remove_columns=['document', 'summary'],
        load_from_cache_file=True

    )
    tokenized_val.save_to_disk(tokenized_val_path)
    logger.info(f"Saved tokenized validation set to {tokenized_val_path}")


logger.info(f"\nPreprocessed Combined Training Dataset:\n{tokenized_train}")
logger.info(f"\nPreprocessed Combined Test Dataset:\n{tokenized_test}")
logger.info(f"\nPreprocessed Combined Validation Dataset:\n{tokenized_val}")

# Inspect a sample
sample = tokenized_train[0]
preprocessing_info = {
    "sample_input_ids_length": len(sample['input_ids']),
    "sample_attention_mask_length": len(sample['attention_mask']),
    "sample_labels_length": len(sample['labels'])
}

logger.info("\nSample preprocessed data:")
logger.info(f"input_ids length: {len(sample['input_ids'])}")
logger.info(f"attention_mask length: {len(sample['attention_mask'])}")
logger.info(f"labels length: {len(sample['labels'])}")

save_metrics_to_json(preprocessing_info, "preprocessing_info.json")

# ============================================================================
# 10. DATA COLLATOR
# ============================================================================

logger.info("\n" + "="*80)
logger.info("SETTING UP DATA COLLATOR")
logger.info("="*80)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
logger.info("Data collator initialized successfully")

# ============================================================================
# 11. EVALUATION METRICS
# ============================================================================

logger.info("\n" + "="*80)
logger.info("SETTING UP EVALUATION METRICS")
logger.info("="*80)

import evaluate
metric = evaluate.load('rouge')

def compute_metrics(eval_pred):
    """Compute ROUGE metrics for summarization"""
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

logger.info("Metrics function defined successfully")

# ============================================================================
# 12. TRAINING CONFIGURATION
# ============================================================================

logger.info("\n" + "="*80)
logger.info("CONFIGURING TRAINING")
logger.info("="*80)

training_output_dir = os.path.join(output_dir, 'model_checkpoints')

training_args = Seq2SeqTrainingArguments(
    output_dir=training_output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    seed=seed,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True if torch.cuda.is_available() else False,
    report_to="none",
    # IMPORTANT CHANGES:
    logging_steps=10,              # Log training loss every 10 steps (much more frequent)
    logging_first_step=True,       # Log the very first step
    # Remove eval_steps and save_steps since you're using epoch strategy
    # eval_steps=2000,             # Remove this - conflicts with eval_strategy="epoch"
    # save_steps=2000,             # Remove this - conflicts with save_strategy="epoch"
    logging_dir=os.path.join(output_dir, 'logs'),
)

training_config = {
    "output_directory": training_output_dir,
    "num_epochs": training_args.num_train_epochs,
    "batch_size": training_args.per_device_train_batch_size,
    "learning_rate": training_args.learning_rate,
    "fp16": training_args.fp16,
    "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
    "weight_decay": training_args.weight_decay,
    "seed": seed,
    "datasets_combined": "XSum + CNN/DailyMail"
}

logger.info("Training arguments:")
logger.info(f"  Output directory: {training_args.output_dir}")
logger.info(f"  Number of epochs: {training_args.num_train_epochs}")
logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
logger.info(f"  Learning rate: {training_args.learning_rate}")
logger.info(f"  FP16: {training_args.fp16}")

save_metrics_to_json(training_config, "training_configuration.json")

# ============================================================================
# 13. TRAINER SETUP
# ============================================================================

logger.info("\n" + "="*80)
logger.info("INITIALIZING TRAINER")
logger.info("="*80)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

logger.info("Trainer initialized successfully")

# ============================================================================
# 14. TRAINING
# ============================================================================

logger.info("\n" + "="*80)
logger.info("STARTING TRAINING ON COMBINED DATASET")
logger.info("="*80)
logger.info("Note: Training on full combined dataset will take considerable time")
logger.info("Consider using a smaller subset for testing: train_dataset=tokenized_train.select(range(2000))")

# For testing purposes with smaller dataset:
# logger.info("\nUsing subset for demonstration purposes (2000 train, 400 val)")
trainer.train_dataset = tokenized_train.select(range(2000))
trainer.eval_dataset = tokenized_val.select(range(500))
train_result = trainer.train()
training_metrics = train_result.metrics
save_metrics_to_file(training_metrics, "training_results.txt")
save_metrics_to_json(training_metrics, "training_results.json")

# Uncomment to train on full dataset:
# train_result = trainer.train()
# training_metrics = train_result.metrics
# logger.info(f"\nTraining completed!")
# logger.info(f"Training metrics: {training_metrics}")
# save_metrics_to_file(training_metrics, "training_results.txt")
# save_metrics_to_json(training_metrics, "training_results.json")

# ============================================================================
# 15. EVALUATION
# ============================================================================

logger.info("\n" + "="*80)
logger.info("EVALUATION")
logger.info("="*80)

validation_results = trainer.evaluate(eval_dataset=tokenized_val.select(range(500)))
logger.info(f"\nValidation Results:\n{validation_results}")
save_metrics_to_file(validation_results, "validation_results.txt")
save_metrics_to_json(validation_results, "validation_results.json")

test_results = trainer.evaluate(eval_dataset=tokenized_test.select(range(400)))
logger.info(f"\nTest Results:\n{test_results}")
save_metrics_to_file(test_results, "test_results.txt")
save_metrics_to_json(test_results, "test_results.json")

# Plot training history
trainer_state_file = os.path.join(training_output_dir, "checkpoint-750", "trainer_state.json")
if os.path.exists(trainer_state_file):
    plot_training_history(trainer_state_file, figures_dir)

# ============================================================================
# 16. SAVE MODEL
# ============================================================================

logger.info("\n" + "="*80)
logger.info("SAVING MODEL")
logger.info("="*80)

model_save_dir = os.path.join(output_dir, "bart_finetuned_combined_news")
trainer.save_model(model_save_dir)
tokenizer.save_pretrained(model_save_dir)
logger.info(f"Model saved to {model_save_dir}")

# ============================================================================
# 17. INFERENCE AND TESTING
# ============================================================================

logger.info("\n" + "="*80)
logger.info("INFERENCE EXAMPLES")
logger.info("="*80)

# Load the fine-tuned model for inference
summarizer = pipeline('summarization', model='facebook/bart-large-xsum', device=-1)

# Test on sample news articles
test_articles = [
    """
    Scientists have discovered a new species of deep-sea fish in the Mariana Trench. 
    The fish, which has been named Pseudoliparis swirei, was found at a depth of 8,000 meters, 
    making it the deepest living fish ever recorded. The creature has adapted to extreme 
    pressure and complete darkness, with translucent skin and no scales. Researchers from 
    the University of Washington spent five years studying the species using special deep-sea 
    cameras. The discovery provides new insights into how life can survive in the most extreme 
    conditions on Earth. The team believes there may be many more undiscovered species in the 
    deep ocean trenches.
    """,
    
    """
    A major technology company announced today that it will invest $10 billion in artificial 
    intelligence research over the next five years. The investment will focus on developing 
    more advanced language models, computer vision systems, and robotics. The company's CEO 
    stated that AI represents the most significant technological shift since the internet. 
    The funding will create 5,000 new jobs and establish research centers in three countries. 
    Critics have raised concerns about the ethical implications of advanced AI systems, 
    particularly regarding job displacement and privacy issues. The company has pledged to 
    work with governments and ethics boards to ensure responsible AI development.
    """,
    
    """
    Climate scientists have reported that global temperatures reached a new record high 
    last month, marking the warmest October since records began in 1880. The increase is 
    attributed to greenhouse gas emissions and the developing El Niño weather pattern. 
    Arctic sea ice extent hit its second-lowest level for October, while Antarctic ice 
    reached record lows. The World Meteorological Organization warns that 2024 is likely 
    to be among the hottest years on record. Environmental groups are calling for urgent 
    action to reduce carbon emissions and transition to renewable energy sources. Several 
    countries have announced plans to accelerate their climate change mitigation efforts.
    """
]

logger.info("Generating summaries for test articles...\n")

inference_results = []
for i, article in enumerate(test_articles, 1):
    logger.info(f"--- Article {i} ---")
    logger.info(f"Original (first 200 chars):\n{article.strip()[:200]}...\n")
    
    summary = summarizer(article.strip(), max_length=60, min_length=20, do_sample=False)
    generated_summary = summary[0]['summary_text']
    
    logger.info(f"Generated Summary:\n{generated_summary}\n")
    logger.info("-" * 80 + "\n")
    
    inference_results.append({
        "article_id": i,
        "article_preview": article.strip()[:300],
        "generated_summary": generated_summary
    })

save_metrics_to_json({"test_articles": inference_results}, "inference_test_articles.json")

# ============================================================================
# 18. SAMPLE PREDICTIONS FROM COMBINED DATASET
# ============================================================================

logger.info("\n" + "="*80)
logger.info("PREDICTIONS ON COMBINED DATASET SAMPLES")
logger.info("="*80)

# Test on actual combined dataset samples
dataset_predictions = []
for i in range(3):
    doc = test['document'].iloc[i]
    ref_summary = test['summary'].iloc[i]
    
    logger.info(f"\n--- Test Sample {i+1} ---")
    logger.info(f"Document (first 300 chars):\n{doc[:300]}...\n")
    logger.info(f"Reference Summary:\n{ref_summary}\n")
    
    gen_summary = summarizer(doc, max_length=128, min_length=20, do_sample=False)
    generated = gen_summary[0]['summary_text']
    
    logger.info(f"Generated Summary:\n{generated}\n")
    logger.info("-" * 80)
    
    dataset_predictions.append({
        "sample_id": i+1,
        "document_preview": doc[:300],
        "reference_summary": ref_summary,
        "generated_summary": generated
    })

save_metrics_to_json({"dataset_predictions": dataset_predictions}, "combined_dataset_sample_predictions.json")

# ============================================================================
# 19. ROUGE SCORES VISUALIZATION
# ============================================================================

def plot_rouge_scores(scores_dict, title, filename):
    """Plot ROUGE scores as a bar chart"""
    # Fix 1: Account for the 'eval_' prefix
    rouge_metrics = {k: v for k, v in scores_dict.items() if 'rouge' in k.lower()}
    if not rouge_metrics:
        logger.warning("No ROUGE metrics found to plot")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(rouge_metrics.keys()),
        y=list(rouge_metrics.values()),
        marker_color=['#89c2e0', '#d500ff', '#00d5ff', '#ff6b6b'],
        text=[f'{v:.2f}' for v in rouge_metrics.values()],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=f'<b>{title}</b>',
        xaxis_title='<b>Metric</b>',
        yaxis_title='<b>Score</b>',
        template=template,
        height=600,
        width=800,
        showlegend=False
    )
    
    filepath = os.path.join(figures_dir, filename)
    fig.write_html(filepath)
    logger.info(f"Saved ROUGE scores plot: {filepath}")
    
    try:
        fig.write_image(filepath.replace('.html', '.png'))
    except Exception as e:
        logger.warning(f"Could not save PNG: {e}")
    
    

# Plot validation and test scores
if 'validation_results' in locals():
    plot_rouge_scores(validation_results, 'Validation ROUGE Scores', 'validation_rouge_scores.html')

if 'test_results' in locals():
    plot_rouge_scores(test_results, 'Test ROUGE Scores', 'test_rouge_scores.html')

# ============================================================================
# 20. COMPARISON VISUALIZATION
# ============================================================================

def plot_summary_comparison(reference, generated, document_preview, sample_id):
    """Create a visual comparison of reference and generated summaries"""
    
    fig = go.Figure()
    
    comparison_text = f"""
    <b>Document Preview:</b><br>
    {document_preview[:300]}...<br><br>
    
    <b>Reference Summary:</b><br>
    {reference}<br><br>
    
    <b>Generated Summary:</b><br>
    {generated}
    """
    
    fig.add_annotation(
        text=comparison_text,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=12),
        align="left",
        xanchor="center",
        yanchor="middle"
    )
    
    fig.update_layout(
        title=f'<b>Summary Comparison - Sample {sample_id}</b>',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template=template,
        height=600,
        width=1000
    )
    
    filename = f"summary_comparison_sample_{sample_id}.html"
    filepath = os.path.join(figures_dir, filename)
    fig.write_html(filepath)
    logger.info(f"Saved comparison plot: {filepath}")
    
    try:
        fig.write_image(filepath.replace('.html', '.png'))
    except Exception as e:
        logger.warning(f"Could not save PNG: {e}")

# Generate comparison visualizations
for pred in dataset_predictions:
    plot_summary_comparison(
        reference=pred['reference_summary'],
        generated=pred['generated_summary'],
        document_preview=pred['document_preview'],
        sample_id=pred['sample_id']
    )

# ============================================================================
# 21. DATASET COMPARISON VISUALIZATION
# ============================================================================

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
            marker_color='#89c2e0',
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
            marker_color='#d500ff',
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
# 22. SUMMARY REPORT
# ============================================================================

logger.info("\n" + "="*80)
logger.info("GENERATING FINAL SUMMARY REPORT")
logger.info("="*80)

summary_report = f"""
{'='*80}
BART COMBINED NEWS SUMMARIZER - SUMMARY REPORT
XSum + CNN/DailyMail Datasets
{'='*80}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DIRECTORIES:
- Output Directory: {output_dir}
- Figures Directory: {figures_dir}
- Metrics Directory: {metrics_dir}
- Logs Directory: {logs_dir}

DATASET INFORMATION:
XSum Dataset:
- Training samples: {len(xsum_dataset['train']):,}
- Validation samples: {len(xsum_dataset['validation']):,}
- Test samples: {len(xsum_dataset['test']):,}

CNN/DailyMail Dataset:
- Training samples: {len(cnn_dataset['train']):,}
- Validation samples: {len(cnn_dataset['validation']):,}
- Test samples: {len(cnn_dataset['test']):,}

Combined Dataset:
- Training samples: {len(train):,}
- Validation samples: {len(val):,}
- Test samples: {len(test):,}

MODEL:
- Checkpoint: {checkpoint}
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}

TRAINING CONFIGURATION:
- Epochs: {training_args.num_train_epochs}
- Batch Size: {training_args.per_device_train_batch_size}
- Learning Rate: {training_args.learning_rate}
- Device: {device}
- FP16: {training_args.fp16}
- Gradient Accumulation Steps: {training_args.gradient_accumulation_steps}

FILES GENERATED:
- Log file: {log_file}
- Figures: See {figures_dir}/
- Metrics: See {metrics_dir}/
- Model: {model_save_dir}

DATASETS COMBINED:
This model was trained on a combination of XSum and CNN/DailyMail datasets,
providing exposure to both abstractive (XSum) and extractive (CNN/DM) 
summarization styles for improved generalization.

{'='*80}
"""

report_path = os.path.join(output_dir, "summary_report.txt")
with open(report_path, 'w') as f:
    f.write(summary_report)

logger.info(summary_report)
logger.info(f"\nSummary report saved to: {report_path}")

logger.info("\n" + "="*80)
logger.info("SCRIPT COMPLETED SUCCESSFULLY")
logger.info("="*80)
logger.info(f"All outputs saved to: {output_dir}")
logger.info("Combined dataset training enables better generalization across news domains!")
logger.info("="*80)