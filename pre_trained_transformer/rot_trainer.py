"""This code was used to fine-tune the pre-trained ViT on an rotational/flip augmented dataset."""

from huggingface_hub import login
from datasets import load_dataset, Dataset
from transformers import AutoImageProcessor, DefaultDataCollator, AutoModelForImageClassification, TrainingArguments, Trainer, pipeline
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, RandomApply
import torchvision.transforms.v2 as v2
from torch.nn import ModuleList
import evaluate
import numpy as np
import wandb

WRITE_TOKEN = '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'
MODEL_NAME = 'long_rot_model'
login(token=WRITE_TOKEN, write_permission=True)

DATASET_DIRECTORY = '/home/uochuba/custom_transformer/long_trees_dataset'

dataset = load_dataset("imagefolder", data_dir=DATASET_DIRECTORY)

# label the names
labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)

_transforms = Compose([v2.RandomHorizontalFlip(p=0.5), 
                       v2.RandomVerticalFlip(p=0.5)]
                       + [RandomResizedCrop(size), ToTensor(), normalize])

def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

dataset.set_transform(transforms)

data_collator = DefaultDataCollator()

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir=MODEL_NAME,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=150,
    warmup_ratio=0.1,
    logging_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    push_to_hub=True,
    report_to="wandb",
    save_total_limit = 5
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)

trainer.train()

