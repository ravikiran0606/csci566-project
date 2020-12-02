---
layout: post
title: Trigger phrase mining to provide additional supervision to low resource NLP tasks!
---

# Introduction

Named Entity Recognition is a fundamental information extraction task that focuses on extracting entities from a given 
text and classifying them using predefined categories (e.g., Locations, persons etc). Recent advances in NER have focused 
on training neural network models with abundance of labeled data. However, collecting human annotations for NER is very 
expensive and time-consuming especially for new domains like biomedical or scientific literature. Thus, a crucial research
question is how to obtain human supervision in an cost-effective way. 

Text classification is a fundamental NLP problem of assigning a category to the given piece of text such as sentiment 
classification, news classification etc. Though this problem is well explored, all the modern deep learning based approaches
requires huge amounts of labeled data to achieve impressive performance. However, collecting labeled data for certain tasks
in low-resource domains are expensive and highly time consuming. 

Thus, we need a way to provide additional supervision to deep learning models for solving NLP tasks like text classification
and sequence tagging when the amount of labeled data is limited and are very expensive to annotate.

# Problem Formulation

We consider the problem of building the trigger phrase dictionary that can capture the global patterns of important phrases
corresponding to each of the label categories in case of a text classification problem (or) important phrases corresponding
to each of the entity types in case of a sequence tagging problem. Using this extracted trigger dictionary, we may be able
to provide additional supervision to deep learning models by using them as an additional labeling method and doing 
hard-matching of the trigger phrases with the test input to get additional labeling decision.

An entity trigger phrase is defined as a collection of words in a sentence that helps humans to recognize a particular 
entity in the sentence. Similarly, an trigger phrase corresponding to a particular label category helps identifying the 
category corresponding to the given text.

For the text classification task, we use the reproducibility classification dataset provided to us by DARPA in which each
sample contains the scientific paper content and four important claims extracted by Human experts and the label corresponds
to whether that particular paper is reproducible or not. For the sequence tagging task, we use the dataset of scientific
papers which are tagged with various entity types such as p-values, effect sizes, model names, study names etc using a 
BIO encoding scheme.

For the text classification task, \newline
Input: Text classification dataset $D$ with a number of categories $C$. \newline
Output: Trigger dictionary $d$ consisting of list of phrases $p$ corresponding to each the categories $C$. 

For the sequence tagging task, \newline
Input: Sequence tagging dataset $D$ with a number of entity types $E$. \newline
Output: Trigger dictionary $d$ consisting of list of phrases $p$ corresponding to each the entity types $E$. 

# Approach

We now present a framework which can extract the trigger phrases corresponding to each of the label categories in case 
of text classification task (or) extract the trigger phrases corresponding to each of the entity types in case of sequence
 tagging task. 

We consider the text classification task of predicting whether a given scientific paper is reproducible or not.  We have
a labeled dataset of around 886 papers. Each paper is parsed using the AllenAI's Science Parse tool 
\footnote{\url{https://github.com/allenai/science-parse}} and converted from a pdf to a structured format which has a list 
of paper sections and its corresponding text. The labeled dataset also includes a list of 4 important claims that are extracted 
from the paper by Human experts. We aim to extract the global patterns of trigger phrases corresponding to each of the 
label categories (i.e., reproducible vs non-reproducible papers). 

For extracting the trigger phrases in the scientific paper, we first identify the most important claim among the given 
4 claims of the paper. For identifying the most important claim among the given 4 claims of the paper, we train a SciBERT
[4] based text classification model (we refer to this model as \emph{SciBERT-v1}) that takes the given 4 claims as inputs
and predicts whether the paper corresponding to the given 4 claims is reproducible or not. The SciBERT based text 
classification model uses SciBERT as a feature extractor followed by a  BiLSTM + Attention layer and a final linear layer 
with Softmax which predicts the probability corresponding to each of the label categories. Once the \emph{SciBERT-v1} model 
is trained, we can use the attention weight corresponding to the claims as a measure of importance and pick the claim with 
the highest attention weight as the most important claim.

![_config.yml]({{ site.baseurl }}/images/LM_models.png)

Then, we identify the list of candidate phrases from this identified claim to compute the importance scores with respect 
to each of the label categories. For extracting the list of candidate phrases, we perform constituency parsing on the 
identified claim and extract the phrases corresponding to the nodes of the parsed tree. We estimate the importance score 
for each of the candidate phrases using the Sampling and Occlusion (SOC) algorithm [2]. This workflow is illustrated in 
Figure \ref{fig:soc_flow}. In SOC algorithm, to estimate the importance of the phrase $p$ in the given input text $S$ 
with label class $c$, we first sample the context around the phrase $p$ in the input text using a language model trained 
on the training corpus. Then, we mask the phrase $p$ from the input text and measure the difference in the prediction logit 
score of the label class $c$ of the input text $S$ without and with masking the phrase $p$ with a mask token. We perform 
this procedure of sampling and measuring the prediction logit score difference for a number of trials and take the average 
difference in prediction logit score to be the importance score for that phrase $p$ in the input text $S$ with label 
class $c$. For measuring the prediction logit score difference, we need to train another SciBERT model (we refer to this 
model as \emph{SciBERT-v2}) with only the important claim as input and aim to predict whether the paper corresponding to 
this important claim is reproducible or not. For sampling, we train a LSTM based language model on the SciBERT tokenized 
text of the training corpus. The relationship between the two SciBERT classifier models are illustrated in the Figure \ref{fig:lm_models}

![_config.yml]({{ site.baseurl }}/images/soc_flow.png)

Once we have the importance scores for all the candidate phrases for each of the label categories, we choose the top-3 
phrases with highest importance scores to be the trigger phrases corresponding to that label category in the given sample. 
Similarly, we identify the top-3 phrases for all the samples in the training dataset and rank them globally by their 
importance scores with respect to each of the label categories. Finally, we pick the top-k (k is an hyper-parameter 
and we chose k to be 100) phrases for each label category and cluster them based on textual similarity. These clusters 
will represent the final trigger phrase dictionary corresponding to each of the label categories which could provide 
additional supervision to the existing deep learning models.

We also consider the sequence tagging task of identifying the named entities like sample size, p-values, effect size, 
model names from the given scientific paper. We are currently applying this framework to extract the trigger phrases 
corresponding to each entity type.

# Experiments

We evaluated the classifiers \emph{SciBERT-v1} and \emph{SciBERT-v2} on the paper reproducibility classification dataset. 
As the total number of samples in this dataset is relatively small (i.e., 886 samples), we perform 5-fold cross validation 
and report the mean of the performance metric to give a better estimate of the model's performance.The results can be 
found in Table \ref{tab:lm_model_results}. Results show that the \emph{SciBERT-v2} model performs significantly better 
than the \emph{SciBERT-v1} model even though its using only the important claim as input to the model.

|    Model   | Accuracy | Precision | Recall | F1-score |
|:----------:|:--------:|:---------:|:------:|:--------:|
| SciBERT-v1 |   66.59  |   0.644   |  0.611 |   0.612  |
| SciBERT-v2 |   90.85  |   0.869   |  0.934 |   0.900  |

Then, we evaluate the extracted trigger phrases corresponding to each of the label categories (i.e., reproducible vs 
non-reproducible) by manually examining them. We gave the list of extracted phrases corresponding to reproducible and 
non-reproducible papers to the domain experts and asked them to evaluate if those phrases make sense intuitively. Some 
of the extracted trigger phrases are shown in the Table \ref{tab:lm_phrases}.

We observed some interesting patterns in the trigger phrases extracted by our framework. The model is able to identify 
the phrases which contain terms like statistical tests having high significance, p-values less than a specific range 
(i.e., p < 0.001) for reproducible papers. For non-reproducible papers, it identifies phrases that contain terms like 
having a difference in test results, incompatibility in experiments and p-values above a specific range (i.e., p > 0.5) etc.

To measure if the p-values play a important role in deciding the reproducibility of the papers, we further analysed the 
distribution of p-values that are extracted from only the most important phrase identified by our framework for each paper 
and found a clear separation in p-values between the reproducible vs non-reproducible papers as shown in the Figure 
\ref{fig:pv_dist}. This observation indicates our phrase extraction framework could also provide additional capability 
to explain the deep learning models prediction.

# References

[1] Bill Yuchen Lin, Dong-Ho Lee, Frank F. Xu, Ouyu Lan, and Xiang Ren. 2019. AlpacaTag: An active learning-based crowd annotation framework for sequence tagging. In Proceedings of the 57th Annual
Meeting of the Association for Computational Linguistics: System Demonstrations, pages 58–63, Florence, Italy. Association for Computational Linguistics.

[2] Xisen Jin, Zhongyu Wei, Junyi Du, Xiangyang Xue, & Xiang Ren. (2020). Towards Hierarchical Importance Attribution: Explaining Compositional Semantics for Neural Sequence Models.

[3] Marco Tulio Ribeiro and Sameer Singh and Carlos Guestrin (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, August 13-17, 2016 (pp. 1135–1144).

[4] Iz Beltagy, Kyle Lo, & Arman Cohan. (2019). SciBERT: A Pretrained Language Model for Scientific Text.





