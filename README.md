# Probing Language Neutrality of Multilingual Representations
> __Summary of "Revisiting the Language Neutrality of Pre-trained Multilingual Representations through Contrastive Learning"__ 

## 1. Introduction 
- __Language Neutrality__, which is often called language agnostic nature, has been treated as a core concept in the field of multilingual PLMs. Since the pre-trained multilingual representations is projected ​​into the unified vector space regardless of its language, it can represent universal meaning if the representations do not include characteristics of each language.
- It is believed that the language neutrality can be improved by alleviating the subspaces of each language in the vector space(Liboviky et al., 2020; Yang et al., 2021; Choenni and Shoutova, 2020; Xie et al., 2022). To verify this, we analyze how __Contrastive Learning__, which deconstructs the subspaces, affects the PLM's language recognition.

## 2. Probing Method

- A __Probing Classifier__ is used to analyze language discrimination ability of multilingual PLMs. If the PLMs' representations contain few characteristics of each language, which is the status of language neutrality, the classifier will not be able to identify the language. The classifier is trained for the following two probing tasks: __(1) Sentence Identification__ __(2) Paraphrased Token Detection__.  

<p align="center">
  <img src="images/probing.PNG" alt="example image" width="500" height="220"/>
</p>

- We train a series of classifiers using the representations from each layer of the PLMs in order to observe changes according to the layer __(layer-wise probing)__. In addition, we train them using *the summed representations*, including those of previous layers, to check which layer adds 'new' information identifying the languages __(cumulative scoring)__.

- After training, the classifiers are evaluated on the test set in the same setting as the training. We compare the case of a multilingual PLM with that of sentence encoders trained through contrastive learning (using 10k, 50k, and 100k XNLI pairs). We also consider the case when performing __Centering__ or __Principal Component Removal (PCR)__ on the multilingual PLM's representations.

## 3. Evaluation (mBERT-cased)

### 3.1. Sentence Identification

□ __Layer-wise Probing__

<p align="center">
  <img src="images/sentence_identification_mbert.PNG" alt="example image" width="600" height="240"/>
</p>

- As can be seen in the graph above, <ins> __contrastive learning reduces language identification information in the upper layers__  </ins>, causing the classifier to be confused in distinguishing the language. As the model learns more sentences through contrastive learning, this tendency becomes stronger.

□ __Cumulative Scoring__

|Layer|1-1|1-2|1-3|1-4|1-5|1-6|1-7|1-8|1-9|1-10|1-11|1-12|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|cls|0.0|-1.81|-2.54|-6.20|+3.19|+4.20|-2.12|+1.65|+0.92|+0.97|+3.93|+1.10|
|avg|0.0|0.0|-0.11|-0.02|+0.02|-0.09|+0.16|-0.09|-0.13|0.0|0.0|+0.02|
|10k|0.0|0.0|-0.04|-0.03|-0.04|-0.09|+0.03|__-0.35__|__-0.87__|__-1.10__|__-2.73__|__-1.56__|
|50k|0.0|0.0|+0.02|-0.09|+0.03|-0.09|-0.04|__-0.29__|__-0.22__|__-0.87__|__-0.96__|__-1.10__|
|100k|0.0|0.0|+0.02|-0.09|+0.03|-0.09|-0.04|__-0.29__|__-0.20__|__-0.70__|__-1.35__|__-1.12__|

- In the case of models trained through contrastive learning, the ability to distinguish languages decreases as representations from the upper layers (8-12) are added. This is consistent with the layers where performance declines in the layer-wise probing, and <ins> __the representations that have lost language identity in these layers appear to be preventing the model from distinguishing the language__. </ins>

### 3.2. Paraphrased Token Detection

□ __Layer-wise Probing__
 
<p align="center">
  <img src="images/token_detection_mbert.PNG" alt="example image" width="600" height="240"/>
</p>

- The upper layers of all models cannot detect paraphrased tokens. Especially, in the cases of the sentence encoders trained through contrastive learning, this pattern begins from the lower layers. It can be interpreted that <ins> __contrastive learning makes it more difficult to detect a small number of tokens of other languages ​​among a large number of English tokens__ <ins>.

□ __Cumulative Scoring__
  
|Layer|1-1|1-2|1-3|1-4|1-5|1-6|1-7|1-8|1-9|1-10|1-11|1-12|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|w/o tuning|0.0|__+0.02__|__+0.01__|__+0.01__|+0.01|0.0|0.0|0.0|0.0|0.0|0.0|-0.01|
|10k|0.0|__+0.02__|__+0.01__|__+0.01__|0.0|0.0|0.0|0.0|0.0|-0.01|-0.03|-0.04|
|50k|0.0|__+0.01__|__+0.01__|__+0.01__|0.0|+0.01|0.0|-0.01|0.0|0.0|-0.01|0.0|
|100k|0.0|__+0.02__|__+0.01__|__+0.01__|0.0|+0.01|0.0|-0.01|0.0|-0.01|0.0|0.0|

- Unlike the cumulative scoring results for Sentence Identification, high performance starting from the bottom layer is maintained until representation from the final layer is added. Consequently, <ins> __adding the representations from upper layers does not result in a decrease in cumulative scoring performance.__ </ins>
- We would like to interpret these opposing results based on (Tenney et al., 2020) and (Vries et al., 2020). They found that tasks requiring less context information, such as POS tagging or NER, are processed in the lower layers of BERT(including mBERT), while tasks requiring more context information, such as co-reference resolution, are processed in the upper layers. 
- Based on these findings, the Paraphrased Token Detection is a task that requires less context information and relies on representations from lower layers of the model. In that case, even if the representation from the upper layer is added, <ins> __the paraphrased token can be detected by relying on the representations from the lower layers.__ </ins>
- In contrast, the Sentence identification is a sentence-level task that relies on more contextual information from the higher layers. As a result, <ins> __if there is little language information in the representations from the upper layers, the performance in identifying languages deteriorates.__ </ins> Note that the sentence encoders trained through contrastive learning is unable to distinguish language in the upper layers.
## 4. Conclusions

- Contrastive learning can improve the language neutrality of representations from the upper layers.
- Unlike the representations from the upper layer, those from the lower layers are not greatly affected in terms of language neutrality.
- For token-level tasks that rely on lower-layer representations and require less contextual information, contrastive learning may have less impact.

## Citing

```
@article{
   title={BERT Rediscovers the Classical NLP Pipeline},
   author={Ian Tenney, Dipanjan Das, Ellie Pavlick},
   booktitle={Annual Meeting of the Association for Computational Linguistics (ACL)},
   year={2019}
}
@article{
   title={What’s so special about BERT’s layers?},
   author={Wietse de Vries, Andreas van Cranenburgh, Malvina Nissim},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2020}
}
@article{
   title={On the Language Neutrality of Pre-trained Multilingual Representations},
   author={Jindřich Libovický, Rudolf Rosa, Alexander Fraser},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2020}
}
@article{
   title={Universal Sentence Representation Learning with Conditional Masked Language Model},
   author={Ziyi Yang, Yinfei Yang, Daniel Cer, Jax Law, Eric Darve},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2021}
}
@article{
   title={Discovering Low-rank Subspaces for Language-agnostic Multilingual Representations},
   author={Zhihui Xie, Handong Zhao, Tong Yu, Shuai Li},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2022}
}
@article{
 title={What does it mean to be language-agnostic? Probing multilingual sentence encoders for typological properties},
 author={Rochelle Choenni, Ekaterina Shutova},
 journal={arXiv preprint arXiv:2009.12862v1},
 year={2020}
}
```
