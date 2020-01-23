# Movie-Review
## Overview:
In this project, we aim to build a classifier to predict the sentiment of a movie review, if it is positive or negative based on a dataset of IMDB movie reviews. Raw labled movie review as the input, the model will output a prediction and predict the sentiment on the test reviews.

# R
## Customized vocabulary:
1. Remove punctuation, convert to lowercase, tokenize

2. Removed the English stopwords set from
NLTK, Python, which have 179 words;  

3. Built the vocabulary based on the third splits and turned in to Document-Term matrix using up to
4-grams and Pruned the vocabulary;

4. Applied two-sample t-test for discriminant analysis to reduce the vocabulary size and ordered the
words by the magnitude of their t-statistics, pick the top 3000 words.  


## Technical details:
Ridge regression with cross-validation was chosen in the project. Preprocessing the text data using step
1-2 in the Customized vocabulary generating step and then matching with the myVocab, the matched
vocab used as the model input.  

I tried random forest and TFidf. Random forest can only reach AUC 0.92, and TFidf can almost reach
AUC 0.957 but with a very large vocabulary size.  

Also, for the vocabulary, I tried to generate three 3000-vocabulary sequence by doing the customized
process on three splits and removed all the numbers and find the common words among three different
vocabulary. The common vocabulary size is 2002, which is supposed to have better performance later on
because it’s more precise and frequent, however it turned out to have worse result than just using the
vocabulary sequence generated by the third split, even though there are some numbers inside.  


## Model validation:
> Performance for Split 1  | 0.9643  
> Performance for Split 2  | 0.9636  
> Performance for Split 3  | 0.9610  
> Vocab Size               | 3000


## Model limitation:
There are still some useless and meaningless words in myVocab, which can be replace with more helpful
words or remove.

##  Future steps:
1. Try to remove all non-letters elements beforehand and some other textclean techniques to obtain
cleaner text data.
2. Try some deep machine learning algorithms, such as LTSM, RNN.
