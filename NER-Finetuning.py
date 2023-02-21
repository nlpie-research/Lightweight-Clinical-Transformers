import transformers as ts
import datasets as ds
from datasets import Dataset
from datasets import load_dataset

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_metric
from transformers import DataCollatorForTokenClassification

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from minialbert_modeling import *

from seqeval.metrics import f1_score, precision_score, recall_score

import numpy as np
import pandas as pd
import math
import csv

tokenizerPath = "bert-base-cased" #Use other tokenizers e.g. 'bert-base-uncased' based on your target models.
tokenizer = ts.AutoTokenizer.from_pretrained(tokenizerPath)

class DatasetInfo:
  def __init__(self, name,
               type="ner", 
               metric=None, 
               load_from_disk=True,
               isMultiSentence=False, 
               validationSubsets=["test"],
               lr=[5e-5, 2e-5, 1e-5], 
               batch_size=[32], 
               epochs=3, 
               runs=1,
               num_labels=None):

    self.name = name
    self.isMultiSentence = isMultiSentence
    self.validationSubsets = validationSubsets
    self.lr = lr
    self.batch_size = batch_size
    self.epochs = epochs
    self.runs = runs
    self.load_from_disk = load_from_disk
    self.type = type
    self.num_labels = num_labels

    if metric == None:
      self.metric = "accuracy"
    else:
      self.metric = metric

    self.fullName = name + "-" + self.metric

class ModelInfo:
  def __init__(self, pretrainedPath, modelPath, isCustom=False, isAdapterTuning=False, use_token_type_ids=True):
    self.pretrainedPath = pretrainedPath
    self.modelPath = modelPath

    self.logsPath = pretrainedPath + f"/"

    self.isCustom = isCustom
    self.isAdapterTuning = isAdapterTuning
    self.use_token_type_ids = use_token_type_ids

  def get_logs_path(self, datasetName):
    return self.logsPath + f"{datasetName}.txt" if not self.isAdapterTuning else self.logsPath + f"{datasetName}-adapter.txt"
  
  def load_model(self, num_labels, ds):
    if self.isCustom:
      if ds.type == "classification":
        model = MiniAlbertForSequenceClassification.from_pretrained(self.modelPath, num_labels=num_labels)
      elif ds.type == "ner":
        model = MiniAlbertForTokenClassification.from_pretrained(self.modelPath, num_labels=num_labels)

      if self.isAdapterTuning:
        model.trainAdaptersOnly()
    else:
      if ds.type == "classification":
        model = ts.AutoModelForSequenceClassification.from_pretrained(self.modelPath, num_labels=num_labels)
      elif ds.type == "ner":
        model = ts.AutoModelForTokenClassification.from_pretrained(self.modelPath, num_labels=num_labels)
    
    return model


def load_datasets(info):
  """#Dataset Utilities"""
  
  if not info.load_from_disk:
    dataset = load_dataset(info.name)
  else:
    dataset = ds.load_from_disk(info.name)

  if info.type == "classification":
    num_labels = len(set(dataset["train"]["labels"]))
    def mappingFunction(samples, **kargs):
      if info.isMultiSentence:
        outputs = tokenizer(samples[dataset["train"].column_names[0]],
                            samples[dataset["train"].column_names[1]],
                            max_length=512,
                            truncation=True,
                            padding=kargs["padding"])
      else:
        outputs = tokenizer(samples[dataset["train"].column_names[0]],
                            truncation=True,
                            max_length=512,
                            padding=kargs["padding"])

      outputs["labels"] = samples["labels"]

      return outputs
  elif info.type == "ner":
    num_labels = len(dataset["info"][0]["all_ner_tags"])
    def mappingFunction(all_samples_per_split, **kargs):
      tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], is_split_into_words=True, truncation=True, max_length=512, padding=kargs["padding"])  
      total_adjusted_labels = []

      for k in range(0, len(tokenized_samples["input_ids"])):
        prev_wid = -1
        word_ids_list = tokenized_samples.word_ids(batch_index=k)
        existing_label_ids = all_samples_per_split["ner_tags"][k]
        i = -1
        adjusted_label_ids = []

        for wid in word_ids_list:
          if(wid is None):
            adjusted_label_ids.append(-100)
          elif(wid!=prev_wid):
            i = i + 1
            adjusted_label_ids.append(existing_label_ids[i])
            prev_wid = wid
          else:
            adjusted_label_ids.append(existing_label_ids[i])
            
        total_adjusted_labels.append(adjusted_label_ids)

      tokenized_samples["labels"] = total_adjusted_labels
      
      return tokenized_samples

  tokenizedTrainDataset = dataset["train"].map(mappingFunction,
                                              batched=True,
                                              remove_columns=dataset["train"].column_names,
                                              fn_kwargs={"padding": "do_not_pad"})
    
  tokenizedValDatasets = []

  for name in info.validationSubsets:
    tokenizedValDataset = dataset[name].map(mappingFunction,
                                            batched=True,
                                            remove_columns=dataset[name].column_names,
                                            fn_kwargs={"padding": "do_not_pad"})
    
    tokenizedValDatasets.append(tokenizedValDataset)

  if info.num_labels != None:
    num_labels = info.num_labels

  return tokenizedTrainDataset, tokenizedValDatasets, num_labels, dataset["info"][0]["all_ner_tags"]


def evaluate(model, info, valDataset, data_collator, all_labels=None):
  model.eval()

  if len(info.metric.split("-")) == 2:
    metric = load_metric(info.metric.split("-")[0])
  else:
    metric = load_metric(info.metric)

  predictions = []
  references = []

  for index, row in enumerate(valDataset):
      sample = data_collator([row])

      for key, value in sample.items():
        sample[key] = value.cuda()

      output = np.argmax(model(**sample).logits.cpu().detach().numpy(), axis=-1)

      if info.type == "ner":
        predictions.append([])
        references.append([])
        for prediction, label in zip(output.reshape(-1), sample["labels"].cpu().detach().numpy().reshape(-1)):
          if label != -100:
            predictions[-1].append(all_labels[prediction])
            references[-1].append(all_labels[label])
      else:
        predictions += list(output.reshape(-1))
        references += list(sample["labels"].cpu().detach().numpy().reshape(-1))


      if index % 100 == 0:
        print(index)
   
  print(predictions[-1])
  print(references[-1])
 
  if info.type == "ner":
    return f"precision: {precision_score(references, predictions)}\nrecall: {recall_score(references, predictions)}\nf1: {f1_score(references, predictions)}"
  else:
    if len(info.metric.split("-")) == 2:
      return metric.compute(predictions=predictions.reshape(-1), references=references.reshape(-1), average=info.metric.split("-")[-1])
    else:
      return metric.compute(predictions=predictions.reshape(-1), references=references.reshape(-1))


def initLogsFile(path):
  try:
    with open(path, mode="w") as f:
      f.write("")
  except:
    pass


def trainAndEvaluate(modelInfo, dsInfo, trainDataset, valDatasets, num_labels, all_labels=None):
  if dsInfo.type == "ner":
    data_collator = ts.DataCollatorForTokenClassification(tokenizer)
  else:
    data_collator = ts.DataCollatorWithPadding(tokenizer, return_tensors="pt")

  logsPath = modelInfo.get_logs_path(dsInfo.fullName.split("/")[-1] if dsInfo.load_from_disk else dsInfo.fullName)
  initLogsFile(logsPath)

  if not modelInfo.use_token_type_ids:
    trainDataset = trainDataset.remove_columns(["token_type_ids"])

  for lr in dsInfo.lr:
    for batch_size in dsInfo.batch_size:
      for _ in range(dsInfo.runs):
        model = modelInfo.load_model(num_labels, dsInfo)

        trainingArguments = ts.TrainingArguments(
            "output/",
            seed=123,
            logging_steps=250,
            save_steps= 2500,
            num_train_epochs=dsInfo.epochs,
            learning_rate=lr,
            lr_scheduler_type="linear",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
        )

        trainer = ts.Trainer(
            model=model,
            args=trainingArguments,
            train_dataset=trainDataset,
            data_collator=data_collator,
        )

        trainer.train()

        for name, dataset in zip(dsInfo.validationSubsets, valDatasets):
          if not modelInfo.use_token_type_ids:
            dataset = dataset.remove_columns(["token_type_ids"])
          result = evaluate(model, dsInfo, dataset, data_collator, all_labels)
          with open(logsPath, mode="a+") as f:
            f.write(f"---HyperParams---\nBatchsize= {batch_size} Lr= {lr}\n---{name} results---\n{str(result)}\n\n")


datasets = [
    DatasetInfo("ClinicalModels/bio-lm/preprocessing/data/preprocessed_datasets/i2b2-2010", #Use the pre-processing code in BioLM (https://github.com/facebookresearch/bio-lm)
                metric="f1",
                load_from_disk=True,
                type="ner",
                isMultiSentence=False,
                lr=[5e-5, 2e-5, 1e-5],
                epochs=3,
                batch_size=[16],
                runs=1),
    DatasetInfo("ClinicalModels/bio-lm/preprocessing/data/preprocessed_datasets/i2b2-2012", #Use the pre-processing code in BioLM (https://github.com/facebookresearch/bio-lm)
                metric="f1",
                load_from_disk=True,
                type="ner",
                isMultiSentence=False, 
                lr=[5e-5, 2e-5, 1e-5],
                epochs=3, 
                batch_size=[16], 
                runs=1),
    DatasetInfo("ClinicalModels/bio-lm/preprocessing/data/preprocessed_datasets/i2b2-2014", #Use the pre-processing code in BioLM (https://github.com/facebookresearch/bio-lm)
                metric="f1",
                load_from_disk=True,
                type="ner",
                isMultiSentence=False,
                lr=[5e-5, 2e-5, 1e-5],
                epochs=3,
                batch_size=[16],
                runs=1),
]

models = [
    ModelInfo("ClinicalModels/models/DistilClinicalBERT",
              "nlpie/distil-clinicalbert",),
    ModelInfo("ClinicalModels/models/ClinicalDistilBERT",
              "nlpie/clinical-distilbert",
              use_token_type_ids=False,),
    ModelInfo("ClinicalModels/models/TinyClinicalBERT",
              "nlpie/tiny-clinicalbert",),
    ModelInfo("ClinicalModels/models/ClinicalMiniALBERT",
              "nlpie/clinical-miniALBERT-128",
              isCustom=True),
]

for dsInfo in datasets:
  trainDataset, valDatasets, num_labels, all_labels = load_datasets(dsInfo)
  print(trainDataset)
  print(valDatasets)
  print(num_labels)
  for modelInfo in models:
    trainAndEvaluate(modelInfo, dsInfo, trainDataset, valDatasets, num_labels, all_labels)
