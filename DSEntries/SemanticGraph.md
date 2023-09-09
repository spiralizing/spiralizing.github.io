+++
title = "Building a semantic network"
hascode = true
date = Date(2022, 10, 14)
rss = "How to build a semantic graph"

tags = ["python","NLP", "data science", "knowledge graph"]
+++

A **semantic network**, sometimes referred as [knowledge graph](https://en.wikipedia.org/wiki/Knowledge_graph) is a graph $\mathcal{G}(v,e)$ where the vertices (or nodes) represent concepts, entities, events, etc. and the edges represent a relationship between the concepts. This relationship is said to be **semantic** because the way these networks are built is by preserving spatial relationships between words in written language. One of the simplest ways to build one of these networks is to create edges between words that appear in the same sentence or paragraph, with the assumption that these words are related somehow. 

Here we are going to build a semantic network from [Cable News Network (CNN)](https://www.cnn.com/) articles that I downloaded from a [Kaggle dataset](https://www.kaggle.com/datasets/hadasu92/cnn-articles-after-basic-cleaning).

Let's do some imports first
```python 
#dataframes and arrays
import pandas as pd
import numpy as np
#plotting
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')
```
We load the `.csv` file as a data frame and drop the NaNs that might be in it:
```python
    
df_cnn = pd.read_csv('Data/CNN_Articles/CNN_Articels_clean.csv')
#remove nans
df_cnn.dropna(inplace=True)
```
now let's explore a bit of the dataset and its statistics, we look at the columns to see what information is contained in the data frame
```python
df_cnn.columns
```
```plaintext
Index(['Index', 'Author', 'Date published', 'Category', 'Section', 'Url',
       'Headline', 'Description', 'Keywords', 'Second headline',
       'Article text'],
      dtype='object')
```
looks like the articles are classified by categories, let's explore the count for each category
```python
plt.figure(figsize=(10,4))
sns.countplot(df_cnn['Category'])
```
~~~
<div class="container">

    <img class="center" src="/assets/cnn_categ.svg" width="500" height="350">

</div>
~~~
we can see that the number of articles is not uniformly distributed across categories, this could add some bias to our analysis so we will need to make a uniform _sample_ of articles to avoid this. We will catch up on that later, for now let's see what text can we use to extract the information we need to build our semantic network.

The data frame has `Headline`, `Description` and `Article text` as our potential candidates to extract the information, let's explore the length -number of words- for each 
```python
#getting the lengths
art_lenghts = [df_cnn['Article text'].apply(lambda text: len(text.split(' '))) ,
    df_cnn['Headline'].apply(lambda text: len(text.split(' '))) ,
    df_cnn['Description'].apply(lambda text: len(text.split(' ')))
    ]

fig, ax = plt.subplots(1,3, figsize=(16,4))

ax[0].hist(art_lenghts[0])
ax[0].set_title(f'Full text (median {np.median(art_lenghts[0]):.0f}) ')
ax[0].set_xlabel('# words')
ax[0].set_ylabel('Count')

ax[1].hist(art_lenghts[1])
ax[1].set_title(f'Headline (median {np.median(art_lenghts[1]):.0f}) ')
ax[1].set_xlabel('# words')
ax[1].set_ylabel('Count')

ax[2].hist(art_lenghts[2])
ax[2].set_title(f'Description (median {np.median(art_lenghts[2]):.0f}) ')
ax[2].set_xlabel('# words')
ax[2].set_ylabel('Count')
```
~~~
<div class="container">

    <img class="center" src="/assets/semnet_distro.svg" width="500" height="250">

</div>
~~~
for simplicity and trying to avoid any other biases that could be introduced by the data, we are going to use the `Description` that has a median of 26 words. 

Now we focus on the categories we are going to use, since `travel`, `vr` and `style` seem to have few articles, we are going to ignore those categories
```python
#categories to use
use_cat = df_cnn['Category'].unique()[:6]
print(use_cat)
```
```plaintext
['news' 'business' 'health' 'entertainment' 'sport' 'politics']
```
for each category we get a sample of `n_articles = 400`, this number is arbitrary and is close to the number of articles that the category `entertainment` has
```python
new_df = pd.DataFrame(columns=df_cnn.columns)
n_articles = 400
for cat in use_cat:
    #temporal slice of dataframe
    tmp_df = df_cnn[df_cnn['Category'] == cat].copy(deep=False)
    #random choice N articles
    selec_art = np.random.choice(range(tmp_df.shape[0]), n_articles)
    #appending the new dataframe ignoring the index so the new dataframe will have new indexes
    new_df = pd.concat([new_df, tmp_df.iloc[selec_art]], ignore_index=True)
```
now that we have stored the articles we are going to work with in a new data frame `new_df` we can load a **language model** from the [spaCy](https://spacy.io/) library

```python
import spacy
#loading the small version of the  model for the english language
nlp = spacy.load('en_core_web_sm')

```
this model has already been prepared to [identify features](https://spacy.io/usage/linguistic-features) for the English language (and some other languages too) and comes with tools that helps us analyze texts, in our case we are going to use its **named entity recognition** tool which will recognize and tag entities for a given text.

Let's try the entity recognition feature from one of the descriptions in the new data frame we made with the sampled articles, first we pass the text through the model
```python
#passing one description to our nlp model
des = nlp(new_df['Description'][32])
#print the original text
print(des.text)
```
```plaintext
A Monday attack on a Fox News crew reporting near the Ukrainian capital of Kyiv left two of the network's journalists dead and its correspondent severely injured, the channel said on Tuesday.
```
now we import `displacy` to output a _fancy_ tagging of named entities
```python
from spacy import displacy 

#displaying entities of the text
displacy.render(des, style='ent')
```
~~~
<div class="entities" style="line-height: 2.5; direction: ltr">A 
    <mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        Monday
        <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
    </mark>
    attack on a 
    <mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        Fox News
        <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
    </mark>
    crew reporting near the 
    <mark class="entity" style="background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        Ukrainian
        <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">NORP</span>
    </mark>
    capital of 
    <mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        Kyiv
        <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
    </mark>
    left 
    <mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        two
        <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
    </mark>
    of the network\'s journalists dead and its correspondent severely injured, the channel said on 
    <mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        Tuesday
        <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
    </mark>.
</div>
~~~ 

as we can see, the named entities that the model recognizes have different types and spacy already assigns the type to each identity, if we want to extract this entities we can do it by accessing `des.ents` from our processed text `des`.

The fact that named entities are also numbers and dates could add noise to our network so we need to get rid of that type of entities, we can inspect what type of entities our model recognizes by doing 
```python
nlp.get_pipe("ner").labels
```
```plaintext
('CARDINAL',
 'DATE',
 'EVENT',
 'FAC',
 'GPE',
 'LANGUAGE',
 'LAW',
 'LOC',
 'MONEY',
 'NORP',
 'ORDINAL',
 'ORG',
 'PERCENT',
 'PERSON',
 'PRODUCT',
 'QUANTITY',
 'TIME',
 'WORK_OF_ART')
```
now we can select what type of entities we are interested in
```python
ent_type = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT','WORK_OF_ART', 'LAW', 'LANGUAGE']
```
and extract them from each of the descriptions in our data `new_df`
```python
art_ents = []
arts_used_ix = []
catego = []
for ix, text in enumerate(new_df['Description']):
    des = nlp(text)
    if len(des.ents) > 1: #storing only those who have more than 1 entity
        arts_used_ix.append(ix)
        in_ents = []
        #saving its category just in case
        catego.append(new_df['Category'][ix])
        for ent in des.ents:
            if ent.label_ in ent_type:
                in_ents.append(ent.text)
        
        art_ents.append(np.unique(np.array(in_ents)))
```
and now we can see how many entities we have in total and how many unique entities there are
```python
unique_ents = [element for nestedlist in art_ents for element in nestedlist]
all_ent_len = len(unique_ents)
unique_ents = np.unique(np.array(unique_ents))
vocab_len = len(unique_ents)

print(f'There are {all_ent_len} named entities with {vocab_len} unique ones')
```
```plaintext
There are 4277 named entities with 1857 unique ones
```
for simplicity, we are going to map the entities to numbers with a dictionary
```python
word2tag = {}
tag2word = {}
for i, ent in enumerate(unique_ents):
    word2tag[ent] = i
    tag2word[i] = ent
```
now we are going to build our semantic network with the `networkx` package, first we initialize the graph
```python
import networkx as nx

#initializing graph
entG = nx.Graph()

```
now we iterate over the entities that we saved and use them as nodes to create edges between them if they appear in the same piece of text (description), we are going to add **weights** to the edges equal to the number of occurrences the two nodes (words) have in the descriptions

```python
for ents in art_ents:
    if len(ents) > 1:
        for i in range(len(ents)):
            for j in range(i+1, len(ents)):
                #getting the labels for the edges
                v1 = word2tag[ents[i]]
                v2 = word2tag[ents[j]]
                #check if the edge exists
                if entG.has_edge(v1, v2):
                    #if does exist it adds +1 to its weight
                    entG[v1][v2]['weight'] +=1
                else:
                    #if doesn't it creates it
                    entG.add_edge(v1, v2, weight=1)
```
and we visualize the resulting network

```python
plot_options = {"node_size": 10, "with_labels": False, "width": 0.8}

fig, ax = plt.subplots(figsize=(15, 15))
ax.axis("off")
nx.draw_networkx(entG, ax=ax,**plot_options)

```
~~~
<div class="container">

    <img class="center" src="/assets/semnet_fullnet.svg" width="500" height="500">

</div>
~~~
it looks that there are a lot of articles that have entities disconnected from the main component of the network, we will throw these small, isolated components of our network and use only the largest connected component

```python
#finding the largest connected component
large_c = max(nx.connected_components(entG), key=len)
large_c = entG.subgraph(large_c).copy()
#saving original labels and relabeling
old2new = dict(zip(large_c, range(1, len(large_c.nodes))))
new2old = dict(zip(range(1, len(large_c.nodes)), large_c))
large_c = nx.relabel_nodes(large_c, mapping=old2new, copy=True)
```
```python
pos = nx.spring_layout(large_c, iterations=100)

fig, ax = plt.subplots(figsize=(12, 12))
ax.axis("off")
nx.draw_networkx(large_c,pos=pos, ax=ax,**plot_options)

```
~~~
<div class="container">

    <img class="center" src="/assets/semnet_largec.svg" width="500" height="500">

</div>
~~~
now we have a cool looking semantic network, there are lots of type of analysis we can make from a network, like identifying relevant nodes, cliques or communities and other topological features that are non evident or very difficult to identify without constructing the network.

Here we are going to use a community detection algorithm to exemplify its use, from `networkx` we load the [Louvain method](https://en.wikipedia.org/wiki/Louvain_method) which uses [modularity](https://en.wikipedia.org/wiki/Modularity_(networks)) as the feature to optimize while finding the communities, this method has the advantage of using the _weights_ of the edges, so it will be more likely for nodes with a strong edge to be in the same community, 

```python
from networkx.algorithms.community import louvain_communities
#finding communities
parts = louvain_communities(large_c)
```
now we are going to assign some randomly chosen colors to each of the communities
```python
import matplotlib
#getting random colors one for each comunity
col_list = list(matplotlib.colors.cnames.keys())
ncolors = np.random.choice(col_list, len(parts), replace=False)
#assigning the colors
colors = ["" for x in range(large_c.number_of_nodes())]
for i,com in enumerate(parts):
    for node in list(com):
        colors[node-1] = ncolors[i]
```
and plot the resulting graph

```python
fig, ax = plt.subplots(figsize=(12, 12))

nx.draw_networkx(large_c, pos=pos,
    node_size=15,with_labels=False, 
    width=0.5, node_color=colors,
)
ax.axis("off")
fig.set_facecolor('grey')
```
~~~
<div class="container">

    <img class="center" src="/assets/semnet_partition.svg" width="500" height="500">

</div>
~~~

we inspect the communities, show how many nodes each of them has
```python
for nc, m in enumerate(parts):
    print(f'Component {nc} has {len(m)} nodes')
```
```plaintext
Component 0 has 2 nodes
Component 1 has 178 nodes
Component 2 has 159 nodes
Component 3 has 8 nodes
Component 4 has 203 nodes
Component 5 has 58 nodes
Component 6 has 22 nodes
Component 7 has 82 nodes
Component 8 has 79 nodes
Component 9 has 65 nodes
Component 10 has 4 nodes
Component 11 has 7 nodes
Component 12 has 23 nodes
Component 13 has 76 nodes
Component 14 has 15 nodes
Component 15 has 9 nodes
Component 16 has 5 nodes
Component 17 has 12 nodes
Component 18 has 8 nodes
Component 19 has 8 nodes
Component 20 has 75 nodes
Component 21 has 9 nodes
Component 22 has 146 nodes
Component 23 has 32 nodes
Component 24 has 54 nodes
```
to have any idea of what this communities represent we can inspect some of them, it looks like there are at least 3 big communities with more than 100 nodes that probably won't have easily interpretable information so we are going to expect one of the small ones, for example the 14th 
```python
[tag2word[new2old[n]] for n in parts[14]]
```
```plaintext
['All-Star',
 'MVP',
 'Steve Nash',
 'Warriors',
 'Jordan',
 "LeBron James'",
 'Looney Tune-acy',
 'Michael Jordan',
 'Space Jam: A New Legacy',
 'Brooklyn Nets',
 'NBA',
 'Kyrie Irving',
 'Adam Silver',
 'the New Orleans Pelicans',
 'African-American']
```

in this particular case it looks like these nodes are related to each other through _basketball_, they might be not the only nodes that are related to basketball but they definitely share more between each other than the rest of the basketball related nodes according to the community detection analysis.

If you liked this example don't forget to take a look at the [notebook here](https://github.com/spiralizing/WebsiteNotebooks/blob/main/Python/SemanticGraph.ipynb).