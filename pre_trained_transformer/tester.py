"""Code created to run testing framwork and create visualizations for each model."""

from huggingface_hub import login
from datasets import load_dataset, Dataset
from transformers import AutoImageProcessor, DefaultDataCollator, AutoModelForImageClassification, TrainingArguments, Trainer, pipeline
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from torch.nn.functional import softmax
import evaluate
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.manifold import TSNE

# login to huggingface, select model, dataset
WRITE_TOKEN = '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'
MODEL_NAME = "full_aug_model"
DATASET_DIRECTORY = '/home/uochuba/custom_transformer/trees_full'
login(token=WRITE_TOKEN, write_permission=True)
dataset = load_dataset("imagefolder", data_dir=DATASET_DIRECTORY)
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# label the names for conversions
labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

true_labels = []
predicted_labels = []
all_logits = []

for i, sample in enumerate(dataset['test']): # iterate over test set
    image = sample['image'] # jpeg file
    inputs = image_processor(image, return_tensors="pt") # pixel values in tensor
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME) # load model
    with torch.no_grad():
        logits = model(**inputs).logits # extract logits

    predicted_label = logits.argmax(-1).item()
    true_label = sample['label']
    all_logits.append(np.array(logits[0])) # keep track of logits
    
    predicted_labels.append(predicted_label)
    true_labels.append(true_label)
    if i % 25 == 0: 
        print(f'{i}/{len(dataset["test"])} predictions completed.') # output mileston statistice

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)
total_correct = np.sum(true_labels == predicted_labels)
all_logits = np.array(all_logits)

# create and save confusion matrix
cf_matrix = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
cf_matrix_df = pd.DataFrame(cf_matrix)
cf_matrix_df.index = cf_matrix_df.columns = ['Grassland\nshrubland', 'Other', 'Plantation', 'Smallholder\nagriculture']
heatmap = sns.heatmap(cf_matrix_df, fmt='', annot=True, cmap='Blues',cbar=True)
heatmap.set_xlabel(f'Predicted\nAccuracy={round(total_correct / len(true_labels), 3)}')
heatmap.set_ylabel('True')
heatmap.set_title(f'{MODEL_NAME} Confusion Matrix')

heatmap.get_figure().savefig(f'{MODEL_NAME}/cf_matrix.png', bbox_inches='tight')

plt.clf()

# create and save confusion matrix normalized over true labels
norm_cf_matrix = confusion_matrix(y_true=true_labels, y_pred=predicted_labels, normalize='true')
norm_cf_matrix_df = pd.DataFrame(norm_cf_matrix)
norm_cf_matrix_df.index = norm_cf_matrix_df.columns = ['Grassland\nshrubland', 'Other', 'Plantation', 'Smallholder\nagriculture']
norm_heatmap = sns.heatmap(np.around(norm_cf_matrix, decimals=3), fmt='', annot=True, cmap='Blues')
norm_heatmap.set_xlabel(f'Predicted\nAccuracy={round(total_correct / len(true_labels), 3)}')
norm_heatmap.set_ylabel('True')
norm_heatmap.set_title(f'{MODEL_NAME} Confusion Matrix')

norm_heatmap.get_figure().savefig(f'{MODEL_NAME}/norm_cf_matrix.png', bbox_inches='tight')

print('Test Accuracy:', total_correct / len(true_labels))

# derive probability distributionpredictions  from logits
p_dist = torch.nn.functional.softmax(torch.tensor(all_logits), dim=1)
p_dist = np.array(p_dist)

num_classes = len(cf_matrix)
stats = []
for i in range(num_classes): # calculate summary statistics
    precision = cf_matrix[i, i] / np.sum(cf_matrix[:, i]) #TP / (TP + FP)
    recall = cf_matrix[i, i] / np.sum(cf_matrix[i, :]) # TP / (TP + FN)
    f1 = 2 * ((precision * recall) / (precision + recall)) # 2 * ((R * P) / (R + P))
    auroc = roc_auc_score(y_true=true_labels, y_score=p_dist, average=None, multi_class='ovr')[i]
    auprc = average_precision_score(y_true=true_labels, y_score=p_dist, average=None)[i]
    stats.append([precision, recall, f1, auroc, auprc])
    
# format and save summary statistics
stats = np.array(stats)
stats_df = pd.DataFrame(stats)
stats_df.index = ['Grassland shrubland', 'Other', 'Plantation', 'Smallholder agriculture']
stats_df.columns = ['Precision', 'Recall', 'F1', 'AUROC', 'AUPRC']
stats_df.loc['Mean'] = stats_df.mean()

stats_df.to_csv(f"{MODEL_NAME}/statistics.csv")  