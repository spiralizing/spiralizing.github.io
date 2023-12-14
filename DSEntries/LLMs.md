+++
title = "Large Language Models Examples"
hascode = true
date = Date(2022, 10, 14)
rss = "Large Language Models Examples"

tags = ["LLM", "data science", "NLP","Fine-tune"]
+++

I know I am a bit late to the *LLM* (and AI) hype, but with so many Language models being released to the public via the [Transformers API by Huggingface](https://huggingface.co/transformers/v3.1.0/index.html) for Python, I decided to give them a try and learn about how to use them not only as assistants but to do research with them or **fine-tune** them.

Here you can find short examples with some of the code I used to explore and learn about these models.

\toc

# Fine tuning for sentiment analysis

Earlier this year, I [made a post](/DSEntries/SentimentSongs1/)  about sentiment analysis of pop song lyrics from different artists. That time I used a lexicon-based analyzer (VADER) that doesn't require any kind of training because it implements a rule-based score. One of the disadvantages of VADER is that is not transferable to other languages because it was built specifically for the English language in social media.

To refresh our memory: sentiment analysis consists in assigning a global *sentiment* (or sentiment score) to a phrase or statement, assigning a category and a weight to each word and averaging them to obtain a sentiment score. Below some examples for the binary (positive or negative) scenario: 

|     Category     |           Example           |
|:----------------:|:---------------------------:|
| Positive emotion |   Love, nice, good, great   |
| Negative emotion | Hurt, ugly, sad, bad, worse |

However, if we want a more complex or custom ruled-based analyzer there are not many other options but to build it ourselves. But don't worry, here is where the *magic* of pre-trained Large Language Models comes into play. 

A pre-trained LLM, it is what it sounds it is: a Language Model that has been trained with a corpus large enough to *learn* non-trivial relationships between words and perform a specific task. Learning these relationships between words allows the model to generate text that resembles text written by someone. But that is not exactly the *magic* I was referring to, what I was referring to is something called **Transfer Learning**, I used this method in a [previous post](/DSEntries/StyleTransfer/) for a different problem. 

The basis of transfer learning is to use the knowledge (representations, relationships) that the model learned during training for a task to perform a new task. In this particular case we want to use a LLM that has been trained for **next token prediction** (text generation) and *fine-tune* it for **sequence classification** (sentiment analysis). The *knowledge* here consists in the relationships between words (tokens) also known as **semantic relationships** that the model *learns* in the **attention layers** of the [Transformer architecture](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)#Architecture). 

Fortunately for us, there are already several pre-trained and fine-tuned models deployed by the community on [Huggingface](https://huggingface.co/models) that can be used for a wide variety of tasks.

Here, we are going to **load, test and fine tune** a model already [fine-tuned to categorize tweets in Spanish.](https://huggingface.co/daveni/twitter-xlm-roberta-emotion-es) This particular model is based on a pre-trained version of [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)) called [XML-ROBERTa](https://huggingface.co/docs/transformers/model_doc/xlm-roberta) that was developed for multi-language purposes. 

Let's start with the code for this entry by loading some libraries: 
```python
#transformers library from huggingface to load the model and tokenizer of the model
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig

#torch utils
import torch.nn as nn
#numpy for numerical and pandas for dataframes
import numpy as np
import pandas as pd
```
And we load [THIS](https://huggingface.co/daveni/twitter-xlm-roberta-emotion-es) model, with it's tokenizer and configuration from the pre-trained version:

```python
#repo adress
model_path = "daveni/twitter-xlm-roberta-emotion-es"
#loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path ) 
#loading configuration
config = AutoConfig.from_pretrained(model_path ) 
#loading the model
model = AutoModelForSequenceClassification.from_pretrained(model_path)
```

One cool thing about open source (and open access) is that we can inspect details about the model, its architecture and parameters:

```python
#inspecting the model
model
```
```plaintext
XLMRobertaForSequenceClassification(
  (roberta): XLMRobertaModel(
    (embeddings): XLMRobertaEmbeddings(
      (word_embeddings): Embedding(250002, 768, padding_idx=1)
      (position_embeddings): Embedding(514, 768, padding_idx=1)
      (token_type_embeddings): Embedding(1, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): XLMRobertaEncoder(
      (layer): ModuleList(
        (0-11): 12 x XLMRobertaLayer(
          (attention): XLMRobertaAttention(
            (self): XLMRobertaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): XLMRobertaSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): XLMRobertaIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): XLMRobertaOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  )
  (classifier): XLMRobertaClassificationHead(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
    (out_proj): Linear(in_features=768, out_features=7, bias=True)
  )
)
```
The first layers are the embedding layers followed by the encoder/decoder layers (12), the output layer and the most important for us: the classifier. As I mentioned previously, we can build custom sentiment analyzers with language models, this classifier has 7 `out_features` or categories: 
* Joy
* Sadness
* Anger
* Fear
* Disgust
* Surprised
* Others 

To be able to use the model let's first define a couple of functions to process and evaluate data:

```python
#we are going to use softmax to normalize the output
from scipy.special import softmax
#function to pre-process tweets
def preprocess_tweet(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

#For evaluation and printing
def get_sentiment(text):
    #tokenize returning tensors for pytorch format
    enc_in = tokenizer(text, return_tensors='pt')
    #passing the encoded input through the model
    output = model(**enc_in)
    #getting score values
    scores = output[0][0].detach().numpy()
    #normalizing with softmax, sorting and printing
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    print(f'Original Text: \n {text} \n')
    for j in range(scores.shape[0]):
        l = config.id2label[ranking[j]]
        s = scores[ranking[j]]
        print(f"{j+1}) {l} {np.round(float(s), 4)}")
    print('===='*20+'\n')
```
Now let's try out the model with the statement: "Me encanta tomar café en la mañana, lamentablemente el día de hoy no he tomado café." Which translates to "I love drinking coffee in the morning, unfortunately I haven't had coffee today".

```python
text = "Me encanta tomar café en la mañana, lamentablemente el día de hoy no he tomado café"
#no need to pre-process this time.
get_sentiment(text)
```
```plaintext
Original Tweet: 
 Me encanta tomar café en las mañanas, lamentablemente el día de hoy no he tomado café. 

1) sadness 0.6812
2) joy 0.1508
3) others 0.1181
4) anger 0.0208
5) surprise 0.0165
6) disgust 0.0074
7) fear 0.0053
================================================================================
```
Which is indeed a sad statement. 

Now let's fine-tune this model, let's say we want to do an analysis of tweets in Spanish with only two sentiments (positive or negative) and for tweets particularly from **México**. First, we need a reliable source of data to train this model: [TASS](http://tass.sepln.org/) is a workshop on semantic analysis that has curated several data sets for semantic analysis in Spanish. They have a [dataset for Mexican Spanish](http://tass.sepln.org/tass_data/download.php?auth=QtDa3s5sA4ReWvYeWrf), I downloaded it and used it to fine-tune this model.

Before loading the data, we are going to define some functions that will help us processing and creating datasets to use with pytorch

```python
import torch
#we define a custom class with torch dataset utilities
class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
      #encodings and labels
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

#using sklearn to split data
from sklearn.model_selection import train_test_split

def get_traintest_datasets(data, target, test_size=0.2):
    train_tweets, test_tweets, train_labels, test_labels = train_test_split(data, target, test_size=test_size)
    train_encodings = tokenizer(train_tweets, padding=True, return_tensors='pt')
    test_encodings = tokenizer(test_tweets,padding=True, return_tensors='pt')
    
    return CustomDataSet(train_encodings, train_labels), CustomDataSet(test_encodings, test_labels)
```

And now we can load our data and build the datasets

```python
#loading training data 
df_tass = pd.read_csv("train_data/train/mx.tsv",sep='\t', names=['id','text','label'])
#dropping nans 
df_tass.dropna(inplace=True)
#extracting values
data = df_tass['text'].values
#preprocessing
data = list(map(lambda x: preprocess(x), data))
#convert N (negative) to 0 and P (positive) to 1
target = df_tass['label'].apply(lambda x: 0 if x=='N' else 1).values

#build datasets
train_dataset, test_dataset = get_traintest_datasets(data, target)
```
We are almost ready to fine-tune our model, the problem now is that our model initially has seven categories and now our training data has only two. We need to modify our model to have two `out_features` on its classifier, in order to do so, we can simply load the same model including two new arguments:

```python
binary_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, ignore_mismatched_sizes=True)
```
```plaintext
Some weights of XLMRobertaForSequenceClassification were not initialized from the model checkpoint at daveni/twitter-xlm-roberta-emotion-es and are newly initialized because the shapes did not match:
- classifier.out_proj.weight: found shape torch.Size([7, 768]) in the checkpoint and torch.Size([2, 768]) in the model instantiated
- classifier.out_proj.bias: found shape torch.Size([7]) in the checkpoint and torch.Size([2]) in the model instantiated
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```
and the transformers package tells us that we need to "train" this model again because we have modified the classifier and its weights have been re-initialized due to a mismatch in dimensions. This is great because is exactly what we wanted, if we inspect the model we can confirm that the last layer has been modified from a seven-category classifier to a binary one:
```python
binary_model
```
```plaintext
XLMRobertaForSequenceClassification(
  (roberta): XLMRobertaModel(
    (embeddings): XLMRobertaEmbeddings(
      (word_embeddings): Embedding(250002, 768, padding_idx=1)
      (position_embeddings): Embedding(514, 768, padding_idx=1)
      (token_type_embeddings): Embedding(1, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): XLMRobertaEncoder(
      (layer): ModuleList(
        (0-11): 12 x XLMRobertaLayer(
          (attention): XLMRobertaAttention(
            (self): XLMRobertaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): XLMRobertaSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): XLMRobertaIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): XLMRobertaOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  )
  (classifier): XLMRobertaClassificationHead(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
    (out_proj): Linear(in_features=768, out_features=2, bias=True)
  )
)
```
Now we can fine-tune our model, there are a couple of alternatives for this, we can either use the [`Trainer`](https://huggingface.co/transformers/v3.1.0/main_classes/trainer.html) included in the Transformers package or we can write our training function with Pytorch like we always do. For this case we are going to use the trainer that comes with the Transformers API because it is very simple and it will save us several lines of code.

To use the trainer we need to load some libraries first

```python
#for evaluation
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
#trainer
from transformers import  Trainer, TrainingArguments
#for metric
import evaluate
#defining the type of metric we are going to use
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return metric.compute(predictions=predictions, references=labels)
```
and define our trainer arguments and the trainer itself

```python
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs (we don't need many)
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=binary_model,                         # the model
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,             # evaluation dataset
    compute_metrics=compute_metrics  # accuracy
)
```
with the trainer we can evaluate first our model, this should give an accuracy of $∼0.50$ if the training data is balanced for a binary case:

```python
trainer.evaluate()
```
```plaintext
{'eval_loss': 0.663949728012085,
 'eval_accuracy': 0.6363636363636364,
 'eval_runtime': 4.9397,
 'eval_samples_per_second': 40.084,
 'eval_steps_per_second': 0.81}
```
which gives us the accuracy of 66 percent, we expect to improve this number after training our model:
```python
trainer.train()
```
```plaintext
{'train_runtime': 374.7152, 
'train_samples_per_second': 6.341, 
'train_steps_per_second': 0.4, 
'train_loss': 0.18583528677622477, 
'epoch': 3.0}
```
and evaluate again

```python
trainer.evaluate()
```
```plaintext
{'eval_loss': 0.6542710065841675,
 'eval_accuracy': 0.8232323232323232,
 'eval_runtime': 5.2173,
 'eval_samples_per_second': 37.951,
 'eval_steps_per_second': 0.767,
 'epoch': 3.0}
```
with a new value for accuracy of 82 percent. This increase in accuracy, although significant it is still not ideal, running for more epochs and/or doing several runnings we can choose the model that performs the best, but that is a problem to explore in another post.

We can finally test our fine-tuned model with new data never seen before

```python
#loading test data
eval_tass = pd.read_csv("train_data/dev/mx.tsv",sep='\t', names=['id','text','label'])
#dropping nans
eval_tass.dropna(inplace=True)
#extracting values
eval_data = eval_tass['text'].values
#preprocessing tweets
eval_data = list(map(lambda x: preprocess(x), eval_data))
#convert N (negative) to 0 and P (positive) to 1
eval_target = eval_tass['label'].apply(lambda x: 0 if x=='N' else 1).values

#encoding the tweets
encoded_eval = list(map(lambda x: tokenizer(x, return_tensors='pt'), eval_data))

#predicting categories
predicted_labels = list(map( lambda x: np.argmax( binary_model(**x)[0][0].detach().numpy()), encoded_eval)) 
```
and use the confusion matrix to visualize the model's performance
```python
cm = confusion_matrix(eval_target, predicted_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=('N','P'))

disp.plot()
```
~~~
<div class="container">

    <img class="center" src="/assets/cm_finetune.svg" width="300" height=300">

</div>
~~~
with it's respective classification report

```python
print(classification_report(eval_target, predicted_labels))
```
```plaintext
Classification report:

                precision    recall  f1-score   support

           0       0.81      0.86      0.84       252
           1       0.86      0.81      0.83       258

    accuracy                           0.83       510
   macro avg       0.83      0.83      0.83       510
weighted avg       0.83      0.83      0.83       510

```
