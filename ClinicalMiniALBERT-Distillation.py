import transformers as ts
from transformers.modeling_outputs import MaskedLMOutput

from datasets import Dataset
from datasets import load_dataset, load_from_disk

import numpy as np
import numpy.core.defchararray as nchar

import torch
import torch.nn as nn
import torch.nn.functional as F

from minialbert_modeling import * #Load it from https://github.com/nlpie-research/MiniALBERT

savePath = "ClinicalModels/models/ClinicalMiniALBERT/"
teacherPath = "emilyalsentzer/Bio_ClinicalBERT"

ds = load_from_disk("tokenizedDatasets/mimic-large/") #Use the pre-processing code availabe in https://github.com/EmilyAlsentzer/clinicalBERT
tokenizer = ts.AutoTokenizer.from_pretrained(teacherPath)

teacher = ts.AutoModelForMaskedLM.from_pretrained(teacherPath)

#Use this code for random initialisation
pretrainedConfig = ts.AutoConfig.from_pretrained("albert-base-v1")
pretrainedConfig.use_adapter = True
pretrainedConfig.num_hidden_layers = 6
pretrainedConfig.reduction_factor = 16
pretrainedConfig.hidden_size = teacher.config.hidden_size
pretrainedConfig.vocab_size = teacher.config.vocab_size
pretrainedConfig.embedding_size = 312

for param in teacher.parameters():
  param.requires_grad = False

class DistillationWrapper(nn.Module):
 def __init__(self, student, teacher):
   super().__init__()

   self.student = student
   self.teacher = teacher

   self.attention_loss = nn.KLDivLoss(reduction="mean")
   self.hidden_loss = nn.CosineEmbeddingLoss(reduction="mean")
   self.output_loss = nn.KLDivLoss(reduction="batchmean")

   self.temperature = 1.0

 def forward(self,
             input_ids,
             attention_mask=None,
             labels=None,
             **kargs):
   
   student_outputs = self.student(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  labels=labels,
                                  output_hidden_states=True,
                                  output_attentions=True,
                                  **kargs)

   with torch.no_grad():
     teacher_outputs = self.teacher(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    output_hidden_states=True,
                                    output_attentions=True,
                                    **kargs)
 
   s_attentions = student_outputs["attentions"]
   t_attentions = [att.detach() for att in teacher_outputs["attentions"]]

   s_embeddings = [student_outputs["hidden_states"][0]]
   t_embeddings = [teacher_outputs["hidden_states"][0].detach()]

   s_hiddens = student_outputs["hidden_states"][1:]
   t_hiddens = [hidden.detach() for hidden in teacher_outputs["hidden_states"][1:]]

   s_logits = student_outputs["logits"]
   t_logits = teacher_outputs["logits"].detach()

   num_of_examples = input_ids.shape[0]
   num_of_heads = s_attentions[0].shape[1]

   lambdas = [1.0] * len(s_attentions)
   
   embedding_loss = self.compute_embedding_loss(s_embeddings, t_embeddings, attention_mask)
   att_loss = self.compute_attention_loss(s_attentions, t_attentions, attention_mask, num_of_heads, lambdas)
   hidden_loss = self.compute_hidden_loss(s_hiddens, t_hiddens, attention_mask, lambdas)
   output_loss = self.compute_output_loss(s_logits, t_logits, labels)

   total_loss = 1.0 * (student_outputs.loss) + 3.0 * (att_loss + hidden_loss) + (5.0 * output_loss)
   #total_loss = 1.0 * (student_outputs.loss) + 3.0 * (embedding_loss + att_loss + hidden_loss) + (5.0 * output_loss) used in the ablation studies

   return MaskedLMOutput(
       loss=total_loss,
       logits=student_outputs.logits,
       hidden_states=student_outputs.hidden_states,
       attentions=student_outputs.attentions,
   )

 def compute_output_loss(self, s_logits, t_logits, labels):
   mask = (labels > -1).unsqueeze(-1).expand_as(s_logits).bool()

   s_logits_slct = torch.masked_select(s_logits, mask)
   s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))
   t_logits_slct = torch.masked_select(t_logits, mask)
   t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))
   assert t_logits_slct.size() == s_logits_slct.size()
   
   output_loss = (
       self.output_loss(
           nn.functional.log_softmax(s_logits_slct / self.temperature, dim=-1),
           nn.functional.softmax(t_logits_slct / self.temperature, dim=-1),
       )
       * (self.temperature) ** 2
   )

   return output_loss

 def compute_attention_loss(self, s_attentions, t_attentions, attention_mask, num_of_heads, lambdas=None):
   total_loss = None

   for index , s_map in enumerate(s_attentions):
     t_map = t_attentions[2*index + 1]

     att_loss_mask = attention_mask.unsqueeze(1)
     att_loss_mask = att_loss_mask.repeat(1 , num_of_heads , 1)
     att_loss_mask = att_loss_mask.unsqueeze(-1).expand_as(s_map).bool()

     s_map_slct = torch.masked_select(s_map, att_loss_mask)
     s_map_slct = s_map_slct.view(-1, s_map.size(-1)) + 1e-12

     t_map_slct = torch.masked_select(t_map, att_loss_mask)
     t_map_slct = t_map_slct.view(-1, t_map.size(-1)) + 1e-12
   
     att_loss = self.attention_loss(torch.log(s_map_slct), t_map_slct)

     if lambdas != None:
       if total_loss == None:
         total_loss = lambdas[index] * att_loss
       else:
         total_loss += lambdas[index] * att_loss
     else:
       if total_loss == None:
         total_loss = att_loss
       else:
         total_loss += att_loss

   return total_loss

 def compute_hidden_loss(self, s_hiddens, t_hiddens, attention_mask, lambdas=None):
   total_loss = None

   for index in range(len(s_hiddens)):
     s_hidden_states = s_hiddens[index]
     t_hidden_states = t_hiddens[2*index + 1]

     mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states).bool()  # (bs, seq_length, dim)

     assert s_hidden_states.size() == t_hidden_states.size()
     
     dim = s_hidden_states.size(-1)

     s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
     s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
     t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
     t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

     target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)

     hidden_loss = self.hidden_loss(s_hidden_states_slct, t_hidden_states_slct, target)

     if lambdas != None:
       if total_loss == None:
         total_loss = lambdas[index] * hidden_loss
       else:
         total_loss += lambdas[index] * hidden_loss
     else:
       if total_loss == None:
         total_loss = hidden_loss
       else:
         total_loss += hidden_loss
   
   return total_loss

 def compute_embedding_loss(self, s_hiddens, t_hiddens, attention_mask):
   total_loss = None

   s_hidden_states = s_hiddens[0]
   t_hidden_states = t_hiddens[0]

   mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states).bool()  # (bs, seq_length, dim)

   assert s_hidden_states.size() == t_hidden_states.size()

   dim = s_hidden_states.size(-1)

   s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
   s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
   t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
   t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

   target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)

   hidden_loss = self.hidden_loss(s_hidden_states_slct, t_hidden_states_slct, target)

   total_loss = hidden_loss

   return total_loss


model = DistillationWrapper(student=student, teacher=teacher)

count = 0

for name , param in model.named_parameters():
  if param.requires_grad == True:
    print(name)
    count += param.numel()

print(count / 1e6)

data_collator = ts.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt")

try:
  with open(savePath + "logs.txt", "w+") as f:
                f.write("")
except:
  pass

class CustomCallback(ts.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)
            with open(savePath + "logs.txt", "a+") as f:
              f.write(str(logs) + "\n")

trainingArguments = ts.TrainingArguments(
    savePath + "checkpoints",
    logging_steps=250,
    overwrite_output_dir=True,
    save_steps=2500,
    num_train_epochs=3,
    learning_rate=5e-4,
    lr_scheduler_type="linear",
    warmup_steps=5000,
    per_gpu_train_batch_size=48, #In our experiments 4 GPUs are used
    weight_decay=1e-4,
    save_total_limit=5,
    remove_unused_columns=True,
)

trainer = ts.Trainer(
    model=model,
    args=trainingArguments,
    train_dataset=ds,
    data_collator=data_collator,
    callbacks=[ts.ProgressCallback(), CustomCallback()],
)

trainer.train()

trainer.save_model(savePath + "final/rawModel/")

def load_and_save_pretrained(model, checkpoint_path, save_path):
  print(model.load_state_dict(torch.load(checkpoint_path)))
  model.student.save_pretrained(save_path)
  return model

load_and_save_pretrained(model, savePath + "final/rawModel/pytorch_model.bin", savePath + "final/model")
