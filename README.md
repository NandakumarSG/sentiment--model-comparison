# Sentiment Model Comparison
This project is to compare the F1 scores on performing sentiment analysis on reviews. The data is obtained from the yelp data set.
https://www.yelp.com/dataset/download

The models that will be compared are the Multinomial Naive Bayes model and the Support Vector Machine. We will be making a comparison between the CountVectorizer and TfIdfVectorizer too.

In order to get faster output and to predict the expected outcome, a smaller datset was used.

The finalized_model.sav is the pickled Naive Bayes Model, trained using CountVectorizer and Yelps' original reviews dataset from the above link. It has 52 million rows and requires lots of processing. I made use of Google's Collaborator: https://colab.research.google.com

I picked the first 0.1 million rows to do the proccesing. I will add the Colab notebook for reference.
