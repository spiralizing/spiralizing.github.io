+++
title = "Building a semantic network"
hascode = true
date = Date(2022, 10, 14)
rss = "How to build a semantic graph"

tags = ["python","NLP", "data science", "knowledge graph"]
+++

A **semantic network**, sometimes referred as [knowledge graph](https://en.wikipedia.org/wiki/Knowledge_graph) is a graph $\mathcal{G}(v,e)$ where the vertices (or nodes) represent concepts, entities, events, etc. and the edges represent a relationship between the concepts. This relationship is said to be **semantic** because in natural language process it is usually defined if the concepts appear in the same sentence, paragraph or document. 
```python 
    #Imports
    import warnings
    import pandas as pd
    import numpy as np
    import spacy
    import seaborn as sns


    import matplotlib.pyplot as plt
    from IPython import display
    display.set_matplotlib_formats('svg')

    nlp = spacy.load('en_core_web_sm')

    warnings.filterwarnings("ignore")
```
We load the `.csv` file as a data frame:
```python
    
df_cnn = pd.read_csv('Data/CNN_Articles/CNN_Articels_clean.csv')
#remove nans
df_cnn.dropna(inplace=True)
```

```python
plt.figure(figsize=(10,4))
sns.countplot(df_cnn['Category'])
```
~~~
<div class="container">

    <img class="center" src="/assets/cnn_categ.svg" width="500" height="350">

</div>
~~~

```python
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

```python
#categories to use
use_cat = df_cnn['Category'].unique()[1:6]
print(use_cat)
```
```plaintext
['business' 'health' 'entertainment' 'sport' 'politics']
```

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
```python
#passing one description to our nlp model
des = nlp(new_df['Description'][32])
#print the original text
print(des.text)
```
```plaintext
A Monday attack on a Fox News crew reporting near the Ukrainian capital of Kyiv left two of the network's journalists dead and its correspondent severely injured, the channel said on Tuesday.
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

```python
ent_type = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT','WORK_OF_ART', 'LAW', 'LANGUAGE']
```

```python
art_ents = []
arts_used_ix = []
catego = []
for ix, text in enumerate(new_df['Description']):
    des = nlp(text)
    if len(des.ents) > 1: #storing only those who have more than 1 entities
        arts_used_ix.append(ix)
        in_ents = []
        catego.append(new_df['Category'][ix])
        for ent in des.ents:
            if ent.label_ in ent_type:
                in_ents.append(ent.text)
        
        art_ents.append(np.unique(np.array(in_ents)))
```

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

```python
word2tag = {}
tag2word = {}
for i, ent in enumerate(unique_ents):
    word2tag[ent] = i
    tag2word[i] = ent
```

```python
import networkx as nx

#initializing graph
entG = nx.Graph()

```

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

```python
from networkx.algorithms.community import louvain_communities
#finding communities
parts = louvain_communities(large_c)
```
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