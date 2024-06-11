# Probing Language Neutrality of Multilingual Representations
> __Title : Revisiting the Language Neutrality of Pre-trained Multilingual Representations through Contrastive Learning__ 

## 1. Introduction 
- __Language Neutrality__, which is often called language agnostic nature, has been treated as a core concept in the field of multilingual PLMs. Since the pre-trained multilingual representations is projected ​​into the unified vector space regardless of its language, it can represent universal meaning if the representations do not include characteristics of each language.
- It is believed that the language neutrality can be improved by alleviating the subspaces of each language in the vector space(Liboviky et al., 2020; Yang et al., 2021; Choenni and Shoutova, 2020; Xie et al., 2024). To verify this, we analyze how __Contrastive Learning__, which aligns the embeddings of each language, affects the PLM's language recognition.

## 2. Probing Method

- A __Probing Classifier__ is used to analyze language discrimination ability of multilingual PLMs. If the PLMs' representations contain few characteristics of each language, which is the status of language neutrality, the classifier will not be able to identify the language. The classifier is trained for the following two probing tasks: __(1) Sentence Identification__ __(2) Paraphrased Token Detection__.  

<그림>

- We train a series of classifiers using the representations from each layer of the PLMs in order to observe changes according to the layer __(layer-wise probing)__. In addition, we train them using *the summed representations*, including those of previous layers, to check which layer adds 'new' information identifying the language __(cumulative scoring)__.

<수식>

- After training, the classifiers are evaluated on the test set in the same setting as the training. We compare the performance of representations from a multilingual PLM with those from sentence encoders trained through contrastive learning (using 10k, 50k, and 100k XNLI pairs). 

## 3. Evaluation

- mBERT(cased)


## 4. Conclusions
