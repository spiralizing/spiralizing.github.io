+++
title = "Gene co-expression in breast cancer"
hascode = true
date = Date(2022, 10, 1)
rss = "Project I worked on during my postdoctoral stay at INMEGEN."

tags = ["science", "research"]
+++


During my postdoctoral stay at the [Computational Genomics Lab](http://csbig.inmegen.gob.mx/) from the [National Institute of Genomic Medicine](https://www.inmegen.gob.mx/) in México City, I worked on developing a hybrid data-driven **clustering** algorithm that combines **eigenvalue decomposition** and **$k$-medoids** to find communities of **genes** with **statistically dependent expression** inside each chromosome. We found that the **statistical dependency** in groups is related to their **physical distance** in the chromosome and this effect is correlated to the **malignancy** of the cancer type. This result confirms previous studies of loss of long-range correlation in gene co-expression and contributes to the understanding of this effect in the intra-chromosome scale.

~~~
<div class="container">

    <img class="center" src="/assets/cancer1.svg" width="500" height="1000">

</div>
~~~

Take a look at the paper [here](https://github.com/spiralizing/CVResume/blob/main/Papers/Paper2021-Cancer.pdf).