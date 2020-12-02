---
layout: post
title: Trigger phrase mining to provide additional supervision to low resource NLP tasks:-
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
and sequence tagging when the amount of labeled data is limited and are very expensive to annotate. The high-level overview of
our approach is illustrated in the below figure, 

![_config.yml]({{ site.baseurl }}/images/trigger_phrase_overview.png)


# Contributions

Our major contributions include,

* Proposed a framework to extract the trigger phrases which can explain the prediction of a deep learning model on really long text
classification task. Proposed a way to apply the Sampling and Occlusion algorithm to long text documents by following a two-stage approach
to first identify the important segment from the input and extract the evidential trigger phrases from the identified important
segment of the input. Previous work on applying the Sampling and Occlusion algorithm (including the original paper) has focused
only on short text classification tasks like sentiment classification and relation classification etc.

* Proposing a framework to extract the trigger phrases which can explain the prediction of a deep learning model on sequence tagging tasks. 
Proposed a way to formulate the named entity recognition task as a entity span classification task and applied the Sampling and Occlusion
algorithm to extract evidential phrases with respect to a particular entity span in the given input sentence. Previous work on providing 
evidence to sequence tagging models such as using LIME algorithm has been focused only on the extraction of evidential word tokens and not phrases.
This framework could be used as a baseline method to accelerate future work on building interpretable sequence tagging models. 


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

For the text classification task, 

Input: Text classification dataset **D** with a number of categories **C**. 

Output: Trigger dictionary **d** consisting of list of phrases **p** corresponding to each the categories **C**. 

For the sequence tagging task, 

Input: Sequence tagging dataset **D** with a number of entity types **E**. 

Output: Trigger dictionary **d** consisting of list of phrases **p** corresponding to each the entity types **E**. 


# Approach

We now present a framework which can extract the trigger phrases corresponding to each of the label categories in case 
of text classification task (or) extract the trigger phrases corresponding to each of the entity types in case of sequence
 tagging task. 

We consider the text classification task of predicting whether a given scientific paper is reproducible or not.  We have
a labeled dataset of around 886 papers. Each paper is parsed using the AllenAI's Science Parse tool 
[https://github.com/allenai/science-parse](https://github.com/allenai/science-parse) and converted from a pdf to a structured 
format which has a list of paper sections and its corresponding text. The labeled dataset also includes a list of 4 important 
claims that are extracted from the paper by Human experts. We aim to extract the global patterns of trigger phrases corresponding 
to each of the label categories (i.e., reproducible vs non-reproducible papers). 

For extracting the trigger phrases in the scientific paper, we first identify the most important claim among the given 
4 claims of the paper. For identifying the most important claim among the given 4 claims of the paper, we train a SciBERT
[4] based text classification model (we refer to this model as *SciBERT-v1*) that takes the given 4 claims as inputs
and predicts whether the paper corresponding to the given 4 claims is reproducible or not. The SciBERT based text 
classification model uses SciBERT as a feature extractor followed by a  BiLSTM + Attention layer and a final linear layer 
with Softmax which predicts the probability corresponding to each of the label categories. Once the *SciBERT-v1* model 
is trained, we can use the attention weight corresponding to the claims as a measure of importance and pick the claim with 
the highest attention weight as the most important claim.

Then, we identify the list of candidate phrases from this identified claim to compute the importance scores with respect 
to each of the label categories. For extracting the list of candidate phrases, we perform constituency parsing on the 
identified claim and extract the phrases corresponding to the nodes of the parsed tree. We estimate the importance score 
for each of the candidate phrases using the Sampling and Occlusion (SOC) algorithm [2]. This workflow is illustrated in 
the below figure, 

![_config.yml]({{ site.baseurl }}/images/soc_flow.png)

In SOC algorithm, to estimate the importance of the phrase **p** in the given input text **S** 
with label class **c**, we first sample the context around the phrase **p** in the input text using a language model trained 
on the training corpus. Then, we mask the phrase **p** from the input text and measure the difference in the prediction logit 
score of the label class **c** of the input text **S** without and with masking the phrase **p** with a mask token. We perform 
this procedure of sampling and measuring the prediction logit score difference for a number of trials and take the average 
difference in prediction logit score to be the importance score for that phrase **p** in the input text **S** with label 
class **c**. For measuring the prediction logit score difference, we need to train another SciBERT model (we refer to this 
model as *SciBERT-v2*) with only the important claim as input and aim to predict whether the paper corresponding to 
this important claim is reproducible or not. For sampling, we train a LSTM based language model on the SciBERT tokenized 
text of the training corpus. The relationship between the two SciBERT classifier models are illustrated in the below figure,

![_config.yml]({{ site.baseurl }}/images/lm_models_both.png)

Once we have the importance scores for all the candidate phrases for each of the label categories, we choose the top-3 
phrases with highest importance scores to be the trigger phrases corresponding to that label category in the given sample. 
Similarly, we identify the top-3 phrases for all the samples in the training dataset and rank them globally by their 
importance scores with respect to each of the label categories. Finally, we pick the top-k (k is an hyper-parameter 
and we chose k to be 100) phrases for each label category and cluster them based on textual similarity. These clusters 
will represent the final trigger phrase dictionary corresponding to each of the label categories which could provide 
additional supervision to the existing deep learning models.

We also consider the sequence tagging task of identifying the named entities like sample size, p-values, effect size, 
model names etc from the given scientific paper. We train the *SciBERT-NER* model on a token level classification task of 
classifying each of the word tokens into one of the entity tags such as B-TN, I-TN, O etc. The SciBERT model operates at
word-piece level and can produce the embeddings for each of the word pieces. So, for classifying each word token, we take
the embedding of the first word-piece token of that word to be the embedding of the entire word token and classify it using 
a linear layer followed by Softmax.

In order to apply the Sampling and Occlusion algorithm directly to the named entity recognition task, we frame the NER task
as entity span classification task by training the SciBERT model with input as \<sentence, entity\> and output as
\<entity_label\>. We refer to this model as *SciBERT-NER-TC* model.  The *SciBERT-NER* and *SciBERT-NER-TC* model architectures are 
illustrated in the below figure,

![_config.yml]({{ site.baseurl }}/images/ner_models.png)

Once we have these trained *SciBERT-NER* and *SciBERT-NER-TC* models, we can extract the trigger phrases corresponding to a
particular entity during the test time. Given a input sentence, we first use the *SciBERT-NER* model to tag the entities on
a token level. Then, we create a sample of the format <input sentence, extracted entity> and use the *SciBERT-NER-TC* model 
and Sampling and Occlusion algorithm to extract evidential phrases from the input sentence part of this artificially created sample.


# Experiments

We evaluated the classifiers *SciBERT-v1* and *SciBERT-v2* on the paper reproducibility classification dataset. 
As the total number of samples in this dataset is relatively small (i.e., 886 samples), we perform 5-fold cross validation 
and report the mean of the performance metric to give a better estimate of the model's performance. The results can be 
found in below table,

![_config.yml]({{ site.baseurl }}/images/results_tc_claim_level.png)

![_config.yml]({{ site.baseurl }}/images/results_tc_paper_level.png)

Results show that the *SciBERT-v2* model performs significantly better 
than the *SciBERT-v1* model even though its using only the important claim as input to the model.

Then, we evaluate the extracted trigger phrases corresponding to each of the label categories (i.e., reproducible vs 
non-reproducible) by manually examining them. We gave the list of extracted phrases corresponding to reproducible and 
non-reproducible papers to the domain experts and asked them to evaluate if those phrases make sense intuitively. Some 
of the extracted trigger phrases are shown in the below table,

![_config.yml]({{ site.baseurl }}/images/examples_tc_claim_level.png)

![_config.yml]({{ site.baseurl }}/images/examples_tc_paper_level.png)

We observed some interesting patterns in the trigger phrases extracted by our framework. The model is able to identify 
the phrases which contain terms like statistical tests having high significance, p-values less than a specific range 
(i.e., p < 0.001) for reproducible papers. For non-reproducible papers, it identifies phrases that contain terms like 
having a difference in test results, incompatibility in experiments and p-values above a specific range (i.e., p > 0.5) etc.

To measure if the p-values play a important role in deciding the reproducibility of the papers, we further analysed the 
distribution of p-values that are extracted from only the most important phrase identified by our framework for each paper 
and found a clear separation in p-values between the reproducible vs non-reproducible papers as shown in the below figure,

![_config.yml]({{ site.baseurl }}/images/pv_dist.png)

This observation indicates our phrase extraction framework could also provide additional capability 
to explain the deep learning models prediction. 

We evaluated the SciBERT-NER model on the feature extraction task of identifying the named entities such as p-value, sample size,
effect size from the input sentence. The results are summarized in the below table, 

![_config.yml]({{ site.baseurl }}/images/results_ner.png)

As we can see from the results, the SciBERT-NER model is able to identify the entities such as model number, study number and p-values
with very high F1 score but not able to identify certain entities such as model name, sample size, sampling method etc.
The entities having a F1 score below 0.90 and having a support of at least 20 occurrences are highlighted in red. These targeted 
entities are explored further and we extracted trigger dictionaries corresponding to those entities so that we can use them 
as additional supervision to improve the performance of the existing SciBERT-NER model.

We evaluated the evidential trigger phrases extracted by the SciBERT-NER-TC model in the given input sentence with respect
to the each of the targeted entities by manually examining them. The model is able to extract the phrases corresponding to
the results of statistical tests, factors and estimates for the entity type "model name" and phrases corresponding to the 
split of the samples, source of the samples for the entity type "sample size" and phrases corresponding to the samples itself
for identifying the entity type "sampling method". These results are in their early stage which could be refined further by 
filtering the candidate phrases in a suitable way. Some of the extracted trigger phrases corresponding to the entity types "model name",
"sample size" and "sampling method" are shown in the below table,                                                       

![_config.yml]({{ site.baseurl }}/images/examples_ner_1.png)
![_config.yml]({{ site.baseurl }}/images/examples_ner_2.png)

# Conclusion and Future work

We presented two frameworks to extract the trigger phrases that can explain the prediction of a deep learning model on two 
types of NLP tasks namely long text classification task and sequence tagging task. These frameworks are model agnostic and
can easily be applied to other deep learning models on similar types of tasks. By extracting the trigger phrases that explain
the model’s prediction, we can find interesting patterns that could be nearly impossible to find otherwise. 

Once these trigger phrases are refined by human experts, we could use them as a way to provide additional supervision to deep
learning models for tasks that does not have enormous labeled data. 

The results of trigger phrase extraction for sequence tagging task can be improved further by couple of refinements. One 
potential improvement is to select the good candidate phrases around the entity spans and compute the importance scores of only
those candidate phrases and pick the top-k phrases. Another improvement can be done by modifying the SOC algorithm to directly
work on the sequence tagging task. 

The Global trigger dictionary is currently build by just aggregating the trigger phrases corresponding to 
the each of the samples in the dataset and use their local importance scores to rank the entire list of trigger phrases 
and pick the top-k phrases for the trigger dictionary. This approach can be further improved by utilizing
the inverse phrase frequency and importance scores normalized by lengths during the ranking stage. The inverse phrase frequency
based method can help us to avoid picking the common phrases as important phrase and normalized importance score based method
provides a fair way to compare the importance scores of phrases having different lengths. 

# References

[1] Bill Yuchen Lin, Dong-Ho Lee, Frank F. Xu, Ouyu Lan, and Xiang Ren. 2019. AlpacaTag: An active learning-based crowd annotation framework for sequence tagging. In Proceedings of the 57th Annual
Meeting of the Association for Computational Linguistics: System Demonstrations, pages 58–63, Florence, Italy. Association for Computational Linguistics.

[2] Xisen Jin, Zhongyu Wei, Junyi Du, Xiangyang Xue, & Xiang Ren. (2020). Towards Hierarchical Importance Attribution: Explaining Compositional Semantics for Neural Sequence Models.

[3] Marco Tulio Ribeiro and Sameer Singh and Carlos Guestrin (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, August 13-17, 2016 (pp. 1135–1144).

[4] Iz Beltagy, Kyle Lo, & Arman Cohan. (2019). SciBERT: A Pretrained Language Model for Scientific Text.





