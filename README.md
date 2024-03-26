# Lightweight Clinical Transformers

Specialised pre-trained language models are becoming more frequent in NLP since they can potentially outperform models trained on generic texts. BioBERT and BioClinicalBERT are two examples of such models that have shown promise in medical NLP tasks. Many of these models are overparametrised and resource-intensive, but thanks to techniques like Knowledge
Distillation (KD), it is possible to create smaller versions that perform almost as well as their larger counterparts. In this work, we specifically focus on development of compact language models for processing clinical texts (i.e. progress notes, discharge summaries etc). 

We developed a number of efficient lightweight clinical transformers using knowledge distillation and continual learning, with the number of parameters ranging from 15 million to 65 million. These models performed comparably to larger models such as BioBERT and ClinicalBioBERT and significantly outperformed other compact models trained on general or biomedical data. 

Our extensive evaluation was done across several standard datasets and covered a wide range of clinical text-mining tasks, including Natural Language Inference, Relation Extraction, Named Entity Recognition, and Sequence Classification. To our knowledge, this is the first comprehensive study specifically focused on creating efficient and compact transformers for clinical NLP tasks. 


## Citation

```bibtex
@article{rohanian2023lightweight,
  title={Lightweight transformers for clinical natural language processing},
  author={Rohanian, Omid and Nouriborji, Mohammadmahdi and Jauncey, Hannah and Kouchaki, Samaneh and Nooralahzadeh, Farhad and Clifton, Lei and Merson, Laura and Clifton, David A and ISARIC Clinical Characterisation Group and others},
  journal={Natural Language Engineering},
  pages={1--28},
  year={2023},
  publisher={Cambridge University Press}
}
```
