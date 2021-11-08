# Book Back Cover: Topic Modeling

 My goal has been to separate books into different topics based on the back cover blurb. 
 The back cover of books often include a description or compelling narrative, but also common are reviews by publishers or other authors.
 The dataset I am using has over 57,000 books and I currently have it divided into 12 topic groups.
 
 ![](https://github.com/DerekCall/NLP/blob/main/countvectorizer-12-topics.png)
 
These topic groups were found using the CountVectorizer tool in scikit-learn Python library. Using the TruncatedSVD tool I was able to reduce the 
dimensionality of the data to 12 topics and then print the top words in each topic category.

Currently I am able to view many seemingly distinct topic groups. The top groups that stand out currently are


