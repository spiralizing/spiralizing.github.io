+++
title = "LLM applications in Academic Libraries"
hascode = true
date = Date(2022, 10, 1)
rss = "Work on AI and LLMs applications in Academic Libraries"

tags = ["libraries", "research", "llm", "AI", "embeddings"]
+++

When I started working for the Libraries at Carnegie Mellon University, part of my research took a new direction. I began working in **Information Science** and **Natural Language Processing**, looking at how **Machine Learning** and **Large Language Models** can help academic libraries. The projects on these subjects are varied, from enriching **catalog metadata** to building **workflows** that are easy to follow and easy to reproduce. The scope and goals of these research projects are supporting and expanding the Libraries mission of data stewardship and knowledge discovery.

\tableofcontents

## Metadata augmentation using NLP, machine learning, and AI chatbots

Libraries spend a lot of time on repetitive tasks like sorting documents into categories and filling in **catalog metadata**, so it is fair to ask how much AI can really help. In this paper we test that on a real academic-library task, classifying documents when only a small amount of data is available, and we compare popular commercial **AI chatbots** against more established machine learning and NLP methods like **XGBoost** and **fine-tuned** models (BERT), evaluating how well both can be integrated in the Libraries environment.

On this task the chatbots performed about the same as one another and did better than the machine learning methods we tested, which is a real advantage when you only have a little local data to train on. They were also easier to work with than writing code, though getting genuinely useful answers was still tricky. One of the main findings in this study is that **carefuly crafted data workflows** can **leverage the capabilities of LLMs** more effectively rather than delegating the full task to the LLM.

~~~
<div class="container">

    <img class="center" src="/assets/llmdh_metadata.png" alt="Comparison of AI chatbots, XGBoost, and BERT fine-tuning for document classification with limited data" width="500" height="350">

</div>
~~~


Read the paper [here](https://www.tandfonline.com/doi/full/10.1080/19386389.2025.2584900).

## An LLM-assisted, human-centered workflow for cataloging

Cataloging rare books takes careful, expert work. To describe a single book you read the object itself, follow strict rules to record its details, check the author against **authority files**, and choose subject terms from standard vocabularies. It is tempting to hand all of this to a **Large Language Model**, but that turns out to be a bad idea. Today's models are not reliable at long, multi-step tasks, they can *make things up*, and the same question can give different answers on different runs, which does not fit a catalog record that needs to be consistent and easy to check.

Instead, we use the model for what it is actually good at. It helps us plan the workflow, look things up across standards and vocabularies, write **code** and **documentation**, and prepare a **draft** that the cataloger then checks and corrects. Each task is sorted by how much human judgment it needs and sent to a person, a simple rule, or a mix of both, while the vocabularies, rules, and local practices a cataloger relies on live in one place the workflow can look up instead of inside the model. It follows five simple goals: keep a person in charge, keep it efficient, and make the results reliable, reproducible, and open. It does not depend on any single model, the code is open source, and as a first test we used it to create **MARC** records for rare books with Carnegie Mellon University Libraries' **Special Collections**.

~~~
<div class="container">

    <img class="center" src="/assets/llmdh_workflow_principles.png" alt="The five design principles of the LLM-assisted cataloging workflow: human-oriented, efficient, reliable, reproducible, sustainable" width="600" height="300">

</div>
~~~


This work was presented as a [poster at Code4Lib 2026](https://doi.org/10.1184/R1/31424834), and a preprint is in preparation. 

## A practical AI-readiness assessment for research data

More and more research data is used to train **Artificial Intelligence**, but data that is ready to publish is not always ready for AI. A dataset can meet every publishing standard and still be a poor fit for training a model. This work makes the case that **FAIR** data (Findable, Accessible, Interoperable, Reusable) is a good starting point but not enough on its own.

It offers a practical way to judge how *AI-ready* a dataset is, based on seven things to look at: how findable and reusable it is, where it came from, how well it is described, whether it was collected ethically, how clearly it can be understood before any model is built, whether it can be maintained over time, and how easily a computer can use it. It then follows the life of a dataset, from collection to release, and points out the stages where bias or errors tend to creep in.

~~~
<div class="container">

    <img class="center" src="/assets/llmdh_aireadiness.png" alt="The seven pre-model dimensions of AI-readiness for research data" width="500" height="500">

</div>
~~~


A preprint is in preparation. In the meantime, you can see the companion [Research Data for AI](https://guides.library.cmu.edu/ResearchDataForAI) guide.

## KiltHub Theses Explorer

One thing I am really interested in is building better ways to organize and explore collections, so that people can discover what a library holds without needing to know the exact words to search for. My research interest here is to design workflows for cataloging and organization that feel intuitive and that help with knowledge exploration and discovery.

The **KiltHub Theses Explorer** is an early experiment in that direction. It is an **interactive map** of CMU theses where you browse by topic instead of by keyword. It uses **embeddings**, which are a way of turning text into numbers that capture its meaning, so theses about similar topics end up close together on the map. The collection becomes a landscape you can wander through. 

~~~
<div class="container">

    <img class="center" src="/assets/llmdh_theses_map.png" alt="Screenshot of the KiltHub Theses Explorer interactive semantic map of CMU theses" width="600" height="380">

</div>
~~~

Explore the [interactive map](https://cmu-lib.github.io/lovedata2025-theses-map/).

## Research data documentation and LLM guides

Good **documentation** is what lets other people reuse data and methods. As part of the Libraries' work I helped write practical guides on how to document the use of **Large Language Models** in research, so those methods stay clear and easy to repeat.

* LLM documentation: [LLM Documentation Guide](https://guides.library.cmu.edu/LLMDocumentationGuide)
* Dataset AI-readiness: [Research Data for AI](https://guides.library.cmu.edu/ResearchDataForAI)
