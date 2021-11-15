# NLP project

NLP Unsupervised Write Up
Topic Model for Books
Derek Call

Abstract

Judging a book by its cover is only a good idea if you use the back cover. The back cover blurbs draw in readers so they need to know more. The genre will be shared and the main characters are introduced with issues that will be addressed in the book. I have done topic modeling to see which words differentiate topics the most. Whether it be a murder mystery thriller, a cookbook, or self-improvement literature I aim to show how natural language processing tools can be leveraged to find patterns in otherwise unlabeled data.  

Design

Book data was collected along with the back cover blurb. I sorted the words so that the most unique words would carry a higher weight, while more common words would be less impactful. I have topics available to review using the pyLDAvis visualization from the sklearn library in python. This shows the separability between the different topics, as well as their similarity.

Data

This dataset was acquired from Kaggle.com (https://www.kaggle.com/jdobrow/57000-books-with-metadata-and-blurbs?select=books_with_blurbs.csv). There are 57,510 books in the dataset with variables for the title, year published, publisher, ISBN, author, and the blurb. This data will lend well to continued work on this project as ISBN numbers are universal so additional data can be found on specific books with ease.

Algorithms

I used python for the data manipulation, text preprocessing, model creation and visualizations.

1.	Using sklearn I was able to fit data on a CountVectorizer instance, a TF-IDF instance, and an LDA instance.
2.	pyLDAvis was used with sklearn for interactive visualizations.




Models
I did many different tests of different vectorizing methods along with hyperparameter tuning of the number of topics, min_df and max_df. The best categorized topic models were when using 12 topics, a min_df of 65and a max_df of 0.05. This was done with a TF-IDF vectorizer so that the most frequent words would be weighted less than rarer words.

Findings

Using and LDA model with the TF-IDF vectorizer I was able to view the top words associated with each of the 12 topics. The topic names that I assigned are:
Topic Names
Mystery/Thriller, Contemporary Fiction, Classic Literature/Fantasy, Self-Improvement, Science/Science Fiction, Cooking, Historical, French, How-To, German, Buzzwords, and Nazi/Horror.

It performed well and made sense on several of the books I manually checked. Two specific books I included in the analysis are The Count of Monte Cristo and Dune. Each book had a high probability assigned to one topic, with the rest being equally unlikely. The Count of Monte Cristo was assigned 87.8% for the Mystery/Thriller topic, while Dune was considered 81.9% a part of the Science/Science Fiction topic.

Tools

•	Pandas, Numpy, and SciKitLearn Python packages were used for data cleaning and modeling

•	pyLDAvis was used for data visualization

