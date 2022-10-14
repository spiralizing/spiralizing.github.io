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