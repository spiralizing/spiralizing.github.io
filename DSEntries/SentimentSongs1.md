+++
title = "Sentiment analysis and topic modeling in pop artists"
hascode = true
date = Date(2022, 10, 14)
rss = "Sentiment analysis and topic model for some pop artists"

tags = ["NLP", "data science", "sentiment analysis","NNMF"]
+++

In Natural Language Processing (**NLP**), sentiment analysis is a methodology to identify and study **_affective states_** in language (text, audio) data. It is usually helpful to analyze ratings or reviews of products but it can be applied to other types of analysis including comments on forums, tweets, etc. 

In this particular case we are going to explore **song lyrics**, and we are going to try to go further and see if we can identify some of the topics associated with particular sentiments.

### How does sentiment analysis work?
The basic idea of sentiment analysis is to assign a label or ranking of a _sentiment_ to a piece of text. The most simple case is to label the text as **positive** or **negative**, some approaches to this problem are based on a _sentiment lexicon_, a list of lexical features (words) labeled as positive or negative:

|     Category     |           Example           |
|:----------------:|:---------------------------:|
| Positive emotion |   Love, nice, good, great   |
| Negative emotion | Hurt, ugly, sad, bad, worse |

Depending on how many _good_ or _bad_ words and taking into account their context in the sample, we can assign a ranking or number to the sentiment. Building an effective sentiment analyzer sounds (and is) complicated, fortunately for us this is a widely explored problem with different approaches in computational linguistics.

We are going to use the **V**alence **A**ware **D**ictionary for s**E**ntiment **R**easoning ([**VADER**](https://ojs.aaai.org/index.php/ICWSM/article/view/14550)), a rule-based model that supposedly outperforms individual human raters. We are going to use **Python** in this case, but VADER has been implemented for different [languages](https://github.com/cjhutto/vaderSentiment).

First, we are going to load the VADER model that is included in [NLTK (Natural Language Tool Kit)](https://www.nltk.org/):

```python
#for DataFrames
import pandas as pd 

#better array management
import numpy as np

#Natural language toolkit
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#sentiment analyzer
sid = SentimentIntensityAnalyzer()
```
Now we can try out the analyzer with an example:
```python
#computing polarity scores
sid.polarity_scores("The last season of game of thrones is the worst of the series")
```
```plaintext
{'neg': 0.255, 'neu': 0.745, 'pos': 0.0, 'compound': -0.6249}
```
The analyzer returns a dictionary with four different quantities: _negative_, _neutral_, _positive_ and _compound_. The first tree are related to the sentiments and the fourth one is a compound score that takes into account all the scores and normalizes the value to be between $[-1,1]$. As we can se, the statement "The last season of game of thrones is the worst of the series" is accurately described as _negative_ with a compound score of $-0.625$.

Now we need the songs from the artists we are going to analyze, I have downloaded [this Kaggle dataset](https://www.kaggle.com/datasets/deepshah16/song-lyrics-dataset) of lyrics from several pop artists and saved it in a folder named `Data/KaggleDS/` and load them as dataframes with pandas:

```python
import os
name_dir = 'Data/KaggleDS/' #directory where the data is stored
name_artists = []
df_artists = []
for file in os.listdir(name_dir):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(name_dir, file))
        df_artists.append(clean_df(df)) #saving the clean dataframes
        name_artists.append(file.split('.')[0]) #saving the names

```
The function `clean_df()` is used to clean the data, you can see more details in the [notebook](https://github.com/spiralizing/WebsiteNotebooks/blob/main/Python/SongLyrics.ipynb) for this code.

Now we check the artists and the number of songs for each artist

```python
n_songs = 0
for i, artist in enumerate(name_artists):
    art_songs = df_artists[i].shape[0]
    print(f'{artist}, {art_songs} lyrics.')
    n_songs += art_songs
    
print(f'\n\n Total: {n_songs} songs from {len(name_artists)} Artists')
```
```plaintext
TaylorSwift, 256 lyrics.
Drake, 307 lyrics.
NickiMinaj, 174 lyrics.
LadyGaga, 156 lyrics.
ColdPlay, 159 lyrics.
PostMalone, 95 lyrics.
SelenaGomez, 85 lyrics.
CardiB, 53 lyrics.
CharliePuth, 49 lyrics.
Rihanna, 215 lyrics.
Maroon5, 97 lyrics.
JustinBieber, 158 lyrics.
Eminem, 366 lyrics.
KatyPerry, 131 lyrics.
EdSheeran, 126 lyrics.
BillieEilish, 52 lyrics.
DuaLipa, 85 lyrics.
ArianaGrande, 164 lyrics.
Khalid, 48 lyrics.
BTS, 233 lyrics.
Beyonce, 163 lyrics.

Total: 3172 songs from 21 Artists
```
and we use the analyzer to compute the compound score and assign it to a new column `sentiment_score` in the dataframe

```python
for df in df_artists:
    df['sentiment_score'] = df['Lyric'].apply(lambda song: sid.polarity_scores(song)['compound']) 
```

Now we can explore, for example how a specific artist changes their sentiment over time
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
plt.title(name_artists[3]) #ladygaga
sns.boxplot(x='Year', y='sentiment_score',data=df_artists[3])

```
~~~
<div class="container">

    <img class="center" src="/assets/ladygaga.svg" width="500" height="450">

</div>
~~~

```python
#defining function
def get_sentplotdata(cleaned_df):
     uyears = np.sort(cleaned_df['Year'].unique())
     med_sscore = [np.median(cleaned_df[cleaned_df['Year'] == year]['sentiment_score'])
          for year in uyears]
     y_errtop = [np.quantile(cleaned_df[cleaned_df['Year'] == year]['sentiment_score'],0.75) for year in uyears]
     y_errbot = [np.quantile(cleaned_df[cleaned_df['Year']==year]['sentiment_score'], 0.25) for year in uyears]
     y_err = [np.subtract(med_sscore,y_errbot),np.subtract(y_errtop, med_sscore)]

     return uyears, med_sscore, y_err

sentiments = [get_sentplotdata(df) for df in df_artists]

plt.figure(figsize=(20,8))
#we choose 5 artists randomly to show
picks = np.random.choice(range(len(name_artists)), 5, replace=False)
for i in picks:
    plt.plot(sentiments[i][0], sentiments[i][1],'s-', label=name_artists[i])
    
plt.ylim(-1.1,1.1)
plt.xlim(2005,2022)
plt.legend()
plt.grid()
plt.xlabel('Year')
plt.ylabel('Sentiment score (median)')

```
bleh

```python
#concatenating all the artists dataframes 
all_df = pd.concat(df_artists)

plt.figure(figsize=(18, 6))
plt.title('Sentiment score per artist')
sns.violinplot(x='Artist', y='sentiment_score',data=all_df)
plt.ylabel('Sentiment score (Compound)')
plt.xlabel('')
plt.xticks(rotation=45)
plt.show()

```