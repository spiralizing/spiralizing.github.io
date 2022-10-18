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

We are going to use the **V**alence **A**ware **D**ictionary for s**E**ntiment **R**easoning ([**VADER**](https://ojs.aaai.org/index.php/ICWSM/article/view/14550)), a rule-based model that supposedly outperforms individual human raters. We are going to use **Python** in this case, but VADER has been implemented for [different languages](https://github.com/cjhutto/vaderSentiment).

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
The analyzer returns a dictionary with four different quantities: _negative_, _neutral_, _positive_ and _compound_. The first tree are related to the sentiments and the fourth one is a compound score that takes into account all the scores and normalizes the value to be between $[-1,1]$. As we can see, the statement "The last season of game of thrones is the worst of the series" is accurately described as _negative_ with a compound score of $-0.625$.

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
the first thing we can notice about the plot is that the sentiment values seem to be very disperse in some of the years, this could be due to few amount of data or the shape of the distribution itself. 

Next, we are going to plot the whole distribution (independent of the year) for every artist, and this time we are going to visualize it as a [violin plot](https://en.wikipedia.org/wiki/Violin_plot) to have a better idea about the shape of the distributions

```python
#concatenating all the artists dataframes 
all_df = pd.concat(df_artists)

plt.figure(figsize=(18, 6))
plt.title('Sentiment score per artist')
#plotting all artists sentiment score distributions and visualizing as violinplots
sns.violinplot(x='Artist', y='sentiment_score',data=all_df)
plt.ylabel('Sentiment score (Compound)')
plt.xlabel('')
plt.xticks(rotation=45)
plt.show()

```
~~~
<div class="container">

    <img class="center" src="/assets/sentiments_artists.svg" width="500" height="450">

</div>
~~~

this plot confirms our suspicion about the shape of the distributions: almost every artist distribution shows [bimodality](https://en.wikipedia.org/wiki/Multimodal_distribution). In this case the bimodality is represented by the two different sentiments indicating that there are few songs across artists that contain a _neutral_ sentiment. This result makes sense, since pop lyrical music usually has strong emotional connotation, with the majority of artists in this sample being _positive_. The only 3 artists with a larger number of _negative songs_ are **Eminem**, **Nicki Minaj** and **Cardi B**, with Eminem being the most _negative_ one. This result is not surprising since Rap (or hip hop) music is well known to talk about social issues and often has violent or _explicit content_ in the lyrics.

### Topic modeling

Now, let's have more fun, let's try to see if there is any relationship between the sentiment and the content (topic) the lyrics have. In other words, we want to see if their positive or negative sentiments are represented by the same or similar words the artist uses.
To explore this question we are going to do **topic modeling** on each artist, so we can assign each of the _songs_ to a particular _topic_ and explore the _most used_ words on each topic.

For the topic model we use the [non-negative matrix factorization (NNMF)](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) method (I will make another entry with details about other topic models). For this particular case, NNMF is more than enough and is easier to understand if you are familiar with linear algebra.\\ 
The basic idea behind NNMF is to factorize an initial matrix $\boldsymbol{V}_{m\times n}$ into a product of two smaller matrices $\boldsymbol{W}_{m \times p} , \boldsymbol{H}_{p \times n}$ :
$$ \boldsymbol{V} \approx \boldsymbol{W} \times \boldsymbol{H} $$
where we can think about what the matrices represent in our problem, the initial matrix would be some type of _normalized frequency_ of each word across all songs ($\boldsymbol{V}_{m\times n}$), where $m$ is the number of words and $n$ the number of songs. In the product one matrix would be the words distributed through topics ($\boldsymbol{W}_{m \times p}$) and the other would be the songs distributed through topics ($\boldsymbol{H}_{p \times n}$) with $p$ being the number of topics, a parameter that we control.

First we are going to construct our _frequency matrix_ or $\boldsymbol{V}$, for this we are going to use a vectorizer from `sklearn` that implements a [term-frequency inverse document frequency (TF-IDF)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) normalization:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
#define the vectorizer
#removing stop words
#max_df = max documents frequency, min_df = minimum documents frequency
tfidf_vec = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')

```
the `max_df` and `min_df`are variables to consider words that appear in at most and at least those fractions or number of documents, next we build our _frequency matrix_, in this example Selena Gomez's one:

```python
# artist on index 6 corresponds to Selena Gomez
freq_mat = tfidf_vec.fit_transform(df_artists[6]['Lyric'])

```
and now we define our NNMF model, we are going to set the number of components (topics) equal to 3 to see if the topics are related to the sentiments. This selection is arbitrary and is under the hypothesis that artists talk about specific topic(s) in a negative or positive way, there are different ways to do this analysis but for simplicity and illustrative purposes we are going to assume that they have only one topic for each sentiment.

```python
from sklearn.decomposition import NMF

nmf_model = NMF(n_components=3)

```
now we fit the model to the data

```python
nmf_model.fit(freq_mat)
```
```plaintext
NMF(n_components=3)
```

and now we can obtain the $\boldsymbol{H}$ matrix that contains the song and the weights for the song on each topic and go further by assigning the number of topic to each of the songs

```python
#get the H matrix
topic_results = nmf_model.transform(freq_mat)
#finding where the largest weight is on every song
#and assigning the song to a topic in the original dataframe of the artist
df_artists[6]['Topic'] = topic_results.argmax(axis=1)
```
now to see what words are the most used on each topic, if we inspect the `nmf_model.components_` entry of our model we will see the matrix $\boldsymbol{W}$ containing the weights for the words on each topic:

```python
nmf_model.components_
```
```plaintext
array([[0.00800022, 0.00368562, 0.01332492, ..., 0.00355086, 0.0046122 ,
        0.00167504],
       [0.        , 0.        , 0.        , ..., 0.        , 0.01988694,
        0.00160906],
       [0.        , 0.        , 0.        , ..., 0.        , 0.00050211,
        0.        ]])
```

with this information we are going to save in the data frame the first 10 most used words on each topic

```python
top_words = []
for ix, topic in enumerate(nmf_model.components_):
    twords = [tfidf_vec.get_feature_names_out()[i]  # topic i
        for i in topic.argsort()[-10:]]
    top_words.append(twords)  

#saving the words in the artist df
df_artists[6]['top_words'] = df_artists[6]['Topic'].apply(lambda topic: top_words[topic])

```
and now we can plot the sentiment distributions for each topic

```python
plt.figure(figsize=(10, 6))
plt.title(name_artists[6])
sns.violinplot(x='Topic', y='sentiment_score', data=df_artists[6])
plt.ylabel('Sentiment score (Compound)')
plt.xlabel('Topic Number')
plt.show()

```

~~~
<div class="container">

    <img class="center" src="/assets/selenag.svg" width="500" height="350">

</div>
~~~

and get the top 10 words for each topic

```python
n_artist = 6
wordsptop = [df_artists[n_artist]
             [df_artists[n_artist]['Topic'] == topic]['top_words'].iloc[0] for topic in range(3)
]
print(f'Topics for artist {name_artists[n_artist]} \n')
for i in range(3):
    print(f'Top 10 words for topic {i}: {wordsptop[i]} \n')

```
```plaintext
Topics for artist SelenaGomez 

Top 10 words for topic 0: ['selena', 'good', 'got', 'want', 'just', 'don', 'know', 'yeah', 'love', 'like'] 

Top 10 words for topic 1: ['gun', 'war', 'blow', 'youre', 'lies', 'head', 'ahead', 'kindness', 'kill', 'em'] 

Top 10 words for topic 2: ['doesn', 'think', 'need', 'baby', 'know', 'magic', 'believe', 'disappear', 'night', 'oh']
```
From this result we can inspect what are the words commonly used in _positive sentiment_ songs, which topic 0 is the one with the most positive median value, words like 'love', 'good' and 'like' with 'selena' makes us think that here she is probably singing about love and relationships or things she likes. The _negative sentiment_ songs seem to talk about 'war' and 'lies'.

Now we do the same for every artist and plot the results
```python
def get_topics(df_artist):
    tfidf_vec = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
    freq_mat = tfidf_vec.fit_transform(df_artist['Lyric'])
    nmf_model = NMF(n_components=3)
    nmf_model.fit(freq_mat)
    top_words = []

    for ix, topic in enumerate(nmf_model.components_):
        twords = [tfidf_vec.get_feature_names_out()[i]  # topic i
                for i in topic.argsort()[-10:]]
        top_words.append(twords)

    # returns the vector of weights for each song on every topic
    topic_results = nmf_model.transform(freq_mat)


    df_artist['Topic'] = topic_results.argmax(axis=1)
    df_artist['top_words'] = df_artist['Topic'].apply(lambda topic: top_words[topic])

    return df_artist

for df in df_artists:
    df = get_topics(df)

fig, axs = plt.subplots(3,7, figsize=(32,14))

for i,ax in enumerate(axs.flatten()):
    sns.violinplot(ax=ax, x='Topic', y='sentiment_score', data=df_artists[i])
    ax.set_title(name_artists[i])
    ax.set_xlabel(' ')
    ax.set_ylabel(' ')
    
```
~~~
<div class="container">

    <img class="center" src="/assets/sent_all_artists.svg" width="500" height="650">

</div>
~~~

and we can inspect the top words for any artist in the same fashion we did for Selena Gomez, for example for Billie Eilish: 

```python
n_artist = 15
wordsptop = [df_artists[n_artist]
             [df_artists[n_artist]['Topic'] == topic]['top_words'].iloc[0] for topic in range(3)
]
print(f'Topics for artist {name_artists[n_artist]} \n')
for i in range(3):
    print(f'Top 10 words for topic {i}: {wordsptop[i]} \n')
```
```plaintext
Topics for artist BillieEilish 

Top 10 words for topic 0: ['bullshit', 'bad', 'wanna', 'leave', 'just', 'way', 'need', 'want', 'know', 'don'] 

Top 10 words for topic 1: ['help', 'say', 'home', 'gonna', 'know', 'just', 'lie', 'love', 'let', 'like'] 

Top 10 words for topic 2: ['sound', 'run', 'town', 'silence', 'favorite', 'make', 'em', 'crown', 'bow', 'watch'] 
```
where the negative topic would be represented by the words on topic 0, the positive is topic 1 and neutral topic is represented by topic 2.

### Conclusion:
Sentiment analysis and topic modeling are two of the most common techniques in natural language processing, here we have combined both to give some insights about the lyrics of some of the top pop artists in the last years, we found that some of the artist seem to sing consistently more positive (or negative) about specific subjects. 

Remember to take a look to the [notebook](https://github.com/spiralizing/WebsiteNotebooks/blob/main/Python/SongLyrics.ipynb) for this example if you want to see more details.