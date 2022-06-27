#!/usr/bin/env python
# coding: utf-8

# # TWITTER SENTIMENT ANALYSIS TO STUDY CUSTOMER BEHAVIOUR

# # ==> Importing necessary packages

# In[1]:


import sys,tweepy,csv,re
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import Stream
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import string


# # AMAZON INDIA

# # ==> Extracting tweets and preprocessing it.

# In[7]:


class SentimentAnalysis:

    def __init__(self):
        self.tweets = []
        self.tweetText = []

    def DownloadData(self):
        # authenticating
        consumerKey = '########################'
        consumerSecret = '#########################################'
        accessToken = '#################################################'
        accessTokenSecret = '##############################################'
        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth)

        # input for term to be searched and how many tweets to search
        searchTerm = input("Enter Keyword/Tag to search about: ")
        NoOfTerms = int(input("Enter how many tweets to search: "))

        # searching for tweets
        self.tweets = tweepy.Cursor(api.search, q=searchTerm, lang = "en").items(NoOfTerms)

        # Open/create a file to append data to
        csvFile = open('Amazon.csv', 'a')

        # Use csv writer
        csvWriter = csv.writer(csvFile)


        # creating some variables to store info
        polarity = 0
        positive = 0
        negative = 0



        # iterating through tweets fetched
        for tweet in self.tweets:
            #Append to temp so that we can store in csv later. 
            self.tweetText.append(self.cleanTweet(tweet.text).encode('utf-8'))
            
            analysis = TextBlob(tweet.text)
            
            polarity += analysis.sentiment.polarity  # adding up polarities to find the average later

            
            # adding reaction of how people are reacting to find average later

            if (analysis.sentiment.polarity > 0.3 and analysis.sentiment.polarity <= 0.6):
                positive += 1

            elif (analysis.sentiment.polarity > -0.6 and analysis.sentiment.polarity <= -0.3):
                negative += 1


        # Write to csv and close csv file
        csvWriter.writerow(self.tweetText)
        csvFile.close()

        # finding average of how people are reacting
        positive = self.percentage(positive, NoOfTerms)
        negative = self.percentage(negative, NoOfTerms)


        # finding average reaction
        polarity = polarity / NoOfTerms

        # printing out data
        print("How people are reacting on " + searchTerm + " by analyzing " + str(NoOfTerms) + " tweets.")
        print()
        print("General Report: ")


        if (polarity > 0.3 and polarity <= 0.6):
            print("Positive")
        elif (polarity > -0.6 and polarity <= -0.3):
            print("Negative")

        print()
        print("Detailed Report: ")
        print(str(positive) + "% people thought it was positive")
        print(str(negative) + "% people thought it was negative")


        self.plotPieChart(positive,negative,searchTerm, NoOfTerms)


    def cleanTweet(self, tweet):
        # Remove Links, Special Characters etc from tweet
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", (tweet)).split())

    # function to calculate percentage
    def percentage(self, part, whole):
        temp = 100 * float(part) / float(whole)
        return format(temp, '.2f')

    def plotPieChart(self, positive,negative,searchTerm, noOfSearchTerms):
        labels = ['Positive [' + str(positive) + '%]',
                  'Negative [' + str(negative) + '%]']
        sizes = [positive, negative]
        colors = ['yellowgreen','red']
        patches, texts = plt.pie(sizes, colors=colors, startangle=90)
        plt.legend(patches, labels, loc="best")
        plt.title('How people are reacting on ' + searchTerm + ' by analyzing ' + str(noOfSearchTerms) + ' Tweets.')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()



if __name__== "__main__":
    sa = SentimentAnalysis()
    sa.DownloadData()


# # ==> Making a word cloud for AMAZON

# In[153]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[154]:


df = pd.read_csv('Amazon.csv', header=None)  #sep='delimiter',


# In[155]:


df.head()


# In[156]:


df = df.T


# In[157]:


df.head()


# In[158]:


#Treating Unicode character
import ast  #AST ..... Abstract Syntax Tree
            #ast. literal_eval: Safely evaluate an expression node or a string 
            #containing a Python literal or container display. 
df[0]= df[0].apply(ast.literal_eval).str.decode("utf-8")
df[0].replace(u"\ufffd", "?")


# In[159]:


#Cleaning the data (Removing Links, Re-Tweet, and other character)
words = df[0].tolist()
words = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +) | (b'RT)", "", str(words))
words = re.sub("(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})", " ", words)
words = re.sub(r':', '',words)
words = re.sub(r'‚Ä¶', '', words)
words = re.sub(r'[^\x00-\x7F]+',' ', words)
words = re.sub(r'[\xe2\x98\x85]+',"", words)


# In[ ]:


words


# In[161]:



words_as_one_string ="".join(words)


# In[ ]:


words_as_one_string


# In[163]:


from wordcloud import WordCloud

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(words_as_one_string))


# In[ ]:





# In[ ]:





# # FLIPKART

# # ==> Extracting tweets and preprocessing it.

# In[ ]:


class SentimentAnalysis:

    def __init__(self):
        self.tweets = []
        self.tweetText = []

    def DownloadData(self):
        # authenticating
        consumerKey = '######################'
        consumerSecret = '######################'
        accessToken = '######################'
        accessTokenSecret = '######################'
        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth)

        # input for term to be searched and how many tweets to search
        searchTerm = input("Enter Keyword/Tag to search about: ")
        NoOfTerms = int(input("Enter how many tweets to search: "))

        # searching for tweets
        self.tweets = tweepy.Cursor(api.search, q=searchTerm, lang = "en").items(NoOfTerms)

        # Open/create a file to append data to
        csvFile = open('Flipkart.csv', 'a')

        # Use csv writer
        csvWriter = csv.writer(csvFile)


        # creating some variables to store info
        polarity = 0
        positive = 0
        negative = 0



        # iterating through tweets fetched
        for tweet in self.tweets:
            #Append to temp so that we can store in csv later.
            self.tweetText.append(self.cleanTweet(tweet.text).encode('utf-8'))
            
            analysis = TextBlob(tweet.text)
            polarity += analysis.sentiment.polarity  # adding up polarities to find the average later

            
            # adding reaction of how people are reacting to find average later

            if (analysis.sentiment.polarity > 0.3 and analysis.sentiment.polarity <= 0.6):
                positive += 1

            elif (analysis.sentiment.polarity > -0.6 and analysis.sentiment.polarity <= -0.3):
                negative += 1


        # Write to csv and close csv file
        csvWriter.writerow(self.tweetText)
        csvFile.close()

        # finding average of how people are reacting
        positive = self.percentage(positive, NoOfTerms)
        negative = self.percentage(negative, NoOfTerms)


        # finding average reaction
        polarity = polarity / NoOfTerms

        # printing out data
        print("How people are reacting on " + searchTerm + " by analyzing " + str(NoOfTerms) + " tweets.")
        print()
        print("General Report: ")


        if (polarity > 0.3 and polarity <= 0.6):
            print("Positive")
        elif (polarity > -0.6 and polarity <= -0.3):
            print("Negative")

        print()
        print("Detailed Report: ")
        print(str(positive) + "% people thought it was positive")
        print(str(negative) + "% people thought it was negative")


        self.plotPieChart(positive,negative,searchTerm, NoOfTerms)


    def cleanTweet(self, tweet):
        # Remove Links, Special Characters etc from tweet
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())

    # function to calculate percentage
    def percentage(self, part, whole):
        temp = 100 * float(part) / float(whole)
        return format(temp, '.2f')

    def plotPieChart(self, positive,negative,searchTerm, noOfSearchTerms):
        labels = ['Positive [' + str(positive) + '%]',
                  'Negative [' + str(negative) + '%]']
        sizes = [positive, negative]
        colors = ['yellowgreen','red']
        patches, texts = plt.pie(sizes, colors=colors, startangle=90)
        plt.legend(patches, labels, loc="best")
        plt.title('How people are reacting on ' + searchTerm + ' by analyzing ' + str(noOfSearchTerms) + ' Tweets.')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()



if __name__== "__main__":
    sa = SentimentAnalysis()
    sa.DownloadData()


# # ==> Making a word cloud for FLIPKART
# 

# In[6]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[9]:


df = pd.read_csv('Flipkart.csv', header =None)


# In[10]:


df.head()


# In[11]:


df = df.T


# In[12]:


df.head()


# In[13]:


#Treating Unicode character
import ast
df[0]= df[0].apply(ast.literal_eval).str.decode("utf-8")
df[0].replace(u"\ufffd", "?")


# In[14]:


#Cleaning the data (Removing Links, Re-Tweet, and other character)
words = df[0].tolist()
words = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +) | (b'RT)", "", str(words))
words = re.sub("(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})", " ", words)
words = re.sub(r':', '',words)
words = re.sub(r'‚Ä¶', '', words)
words = re.sub(r'[^\x00-\x7F]+',' ', words)
words = re.sub(r'[\xe2\x98\x85]+',"", words)


# In[ ]:


words


# In[16]:


words_as_one_string ="".join(words)


# In[ ]:


words_as_one_string


# In[19]:


from wordcloud import WordCloud

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(words_as_one_string))


# In[ ]:





# In[ ]:





# # SNAPDEAL

# # ==> Extracting tweets and preprocessing it.

# In[20]:


class SentimentAnalysis:

    def __init__(self):
        self.tweets = []
        self.tweetText = []

    def DownloadData(self):
        # authenticating
        consumerKey = '######################'
        consumerSecret = '######################'
        accessToken = '######################'
        accessTokenSecret = '######################'
        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth)

        # input for term to be searched and how many tweets to search
        searchTerm = input("Enter Keyword/Tag to search about: ")
        NoOfTerms = int(input("Enter how many tweets to search: "))

        # searching for tweets
        self.tweets = tweepy.Cursor(api.search, q=searchTerm, lang = "en").items(NoOfTerms)

        # Open/create a file to append data to
        csvFile = open('Snapdeal.csv', 'a')

        # Use csv writer
        csvWriter = csv.writer(csvFile)


        # creating some variables to store info
        polarity = 0
        positive = 0
        negative = 0



        # iterating through tweets fetched
        for tweet in self.tweets:
            #Append to temp so that we can store in csv later. 
            self.tweetText.append(self.cleanTweet(tweet.text).encode('utf-8'))
            
            analysis = TextBlob(tweet.text)
           
            polarity += analysis.sentiment.polarity  # adding up polarities to find the average later

            
            # adding reaction of how people are reacting to find average later

            if (analysis.sentiment.polarity > 0.3 and analysis.sentiment.polarity <= 0.6):
                positive += 1

            elif (analysis.sentiment.polarity > -0.6 and analysis.sentiment.polarity <= -0.3):
                negative += 1


        # Write to csv and close csv file
        csvWriter.writerow(self.tweetText)
        csvFile.close()

        # finding average of how people are reacting
        positive = self.percentage(positive, NoOfTerms)
        negative = self.percentage(negative, NoOfTerms)


        # finding average reaction
        polarity = polarity / NoOfTerms

        # printing out data
        print("How people are reacting on " + searchTerm + " by analyzing " + str(NoOfTerms) + " tweets.")
        print()
        print("General Report: ")


        if (polarity > 0.3 and polarity <= 0.6):
            print("Positive")
        elif (polarity > -0.6 and polarity <= -0.3):
            print("Negative")

        print()
        print("Detailed Report: ")
        print(str(positive) + "% people thought it was positive")
        print(str(negative) + "% people thought it was negative")


        self.plotPieChart(positive,negative,searchTerm, NoOfTerms)


    def cleanTweet(self, tweet):
        # Remove Links, Special Characters etc from tweet
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())

    # function to calculate percentage
    def percentage(self, part, whole):
        temp = 100 * float(part) / float(whole)
        return format(temp, '.2f')

    def plotPieChart(self, positive,negative,searchTerm, noOfSearchTerms):
        labels = ['Positive [' + str(positive) + '%]',
                  'Negative [' + str(negative) + '%]']
        sizes = [positive, negative]
        colors = ['yellowgreen','red']
        patches, texts = plt.pie(sizes, colors=colors, startangle=90)
        plt.legend(patches, labels, loc="best")
        plt.title('How people are reacting on ' + searchTerm + ' by analyzing ' + str(noOfSearchTerms) + ' Tweets.')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()



if __name__== "__main__":
    sa = SentimentAnalysis()
    sa.DownloadData()


# # ==> Making a word cloud for SNAPDEAL

# In[10]:


import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization


# In[13]:


df = pd.read_csv('Snapdeal.csv', header =None)


# In[14]:


df.head()


# In[15]:


df = df.T


# In[16]:


df.head()


# In[17]:


import ast
df[0]= df[0].apply(ast.literal_eval).str.decode("utf-8")
df[0].replace(u"\ufffd", "?")


# In[18]:


#Cleaning the data (Removing Links, Re-Tweet, and other character)
words = df[0].tolist()
words = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +) | (b'RT)", "", str(words))
words = re.sub("(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})", " ", words)
words = re.sub(r':', '',words)
words = re.sub(r'‚Ä¶', '', words)
words = re.sub(r'[^\x00-\x7F]+',' ', words)
words = re.sub(r'[\xe2\x98\x85]+',"", words)


# In[ ]:


words


# In[20]:


words_as_one_string ="".join(words)


# In[ ]:


words_as_one_string


# In[22]:


from wordcloud import WordCloud

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(words_as_one_string))


# In[ ]:




