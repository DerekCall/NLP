# Book Back Cover: Topic Modeling

 My goal has been to separate books into different topics based on the back cover blurb. 
 The back cover of books often include a description or compelling narrative, but also commonly include reviews by publishers or other authors.
 
 The dataset I am using has over 57,000 books and I currently have it divided into 12 topic groups.
 
 ![](https://github.com/DerekCall/NLP/blob/main/countvectorizer-12-topics.png)
 
These topic groups were found using the CountVectorizer tool in scikit-learn Python library. Using the TruncatedSVD tool I was able to reduce the 
dimensionality of the data to 12 topics and then print the top words in each topic category.

Currently I am able to view many seemingly distinct topic groups. 

## Some of the groups that stand out currently:
- Spanish (Topic 0)
- German (Topic 1)
- French (topic 2)
- Italian (Topic 5)
- Crime Novels (Topic 8)
- Fantasy Novels (Topic 11)

I believe more analysis  prioritizing the literature in English would be most in line with my project goals. I would like to use pyLDAvis to more readily view
the overlaps in different genres, as well as pick out at least 5 more genres in English. Continually tuning the different hyperparameters of the models used and potentially using different models would yield more actionable results.


