

```python
Observation1: Last 100 tweets reffering to CBS and FOX is mostly Positive.
Observation2: Last 100 tweets reffering to BBC and CNN are  mostly Poitive or neutral sentiments.
Observation3: Last 100 tweets reffering to nytimes are  mostly negetive sentiments.
```


```python
import tweepy
import json
import numpy as np
import pandas as pd
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import time
import seaborn as sns
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from datetime import datetime 
%matplotlib inline
```


```python
# Twitter API Keys
consumer_key = "DcKiE8BJZ2kdqjtKpZPsLMB4c"
consumer_secret = "vtKkC2ieEv8sd0MjKHKWCXH81D5Tko1SWfPUhFoNGiEGAoaxkK"
access_token = "942846023411609601-E0Ydz29HxJjc0GwsWl0mBS7VAzS4MfG"
access_token_secret = "zMUaPpc1NPqbpbQWi5F2dgB4CQdXqYzkx5zRlHKFTqnKA"

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
target_user = ("@nytimes", "@CBS", "@FOX", "@CNN", "@BBC")

sentiments = []
# Loop through each user
for user in target_user:
    counter = 1

    public_tweets = api.user_timeline(user, result_type="recent", count = 100)

    for tweet in public_tweets:
        print(json.dumps(tweet, sort_keys=True, indent=4, separators=(',', ': ')))
        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]

        sentiments.append({"Date": tweet["created_at"],
                                   "Account": tweet['user']['screen_name'],
                                   "Text": tweet["text"],
                                   "Compound": analyzer.polarity_scores(tweet["text"])["compound"],
                                   "Positive": analyzer.polarity_scores(tweet["text"])["pos"],
                                   "Negative": analyzer.polarity_scores(tweet["text"])["neu"],
                                   "Neutral": analyzer.polarity_scores(tweet["text"])["neg"],
                                   "Tweets Ago": counter})
        counter += 1
```


```python
sentiments_pd = pd.DataFrame.from_dict(sentiments)

```


```python
sentiments_pd.head()

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Account</th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Text</th>
      <th>Tweets Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nytimes</td>
      <td>0.7096</td>
      <td>Tue Jan 09 18:33:08 +0000 2018</td>
      <td>0.763</td>
      <td>0.000</td>
      <td>0.237</td>
      <td>In a visit to China, Emmanuel Macron offered t...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nytimes</td>
      <td>0.0000</td>
      <td>Tue Jan 09 18:23:33 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>RT @mairalg: How have @nytimes readers reacted...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nytimes</td>
      <td>0.0000</td>
      <td>Tue Jan 09 18:10:13 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Opinion: A worldwide influenza pandemic is lit...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nytimes</td>
      <td>-0.4215</td>
      <td>Tue Jan 09 18:00:04 +0000 2018</td>
      <td>0.719</td>
      <td>0.191</td>
      <td>0.090</td>
      <td>RT @nytvideo: “When my son told me he wasn’t a...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nytimes</td>
      <td>0.2263</td>
      <td>Tue Jan 09 17:50:48 +0000 2018</td>
      <td>0.899</td>
      <td>0.000</td>
      <td>0.101</td>
      <td>Breaking News: Joe Arpaio, the former Arizona ...</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
sentiments_pd.to_csv("Media_sentiments.csv")
```


```python
plt.figure(figsize=(15,10))
plt.scatter(sentiments_pd[sentiments_pd.Account=='nytimes']['Tweets Ago'],
            sentiments_pd[sentiments_pd.Account=='nytimes']['Compound'],
           label='New York Times', color='yellow', edgecolors='black')
plt.scatter(sentiments_pd[sentiments_pd.Account=='CNN']['Tweets Ago'],
            sentiments_pd[sentiments_pd.Account=='CNN']['Compound'],
           label='CNN',  color='red', edgecolors='black')
plt.scatter(sentiments_pd[sentiments_pd.Account=='BBC']['Tweets Ago'],
            sentiments_pd[sentiments_pd.Account=='BBC']['Compound'],
           label='BBC', color='cyan', edgecolors='black')
plt.scatter(sentiments_pd[sentiments_pd.Account=='fox']['Tweets Ago'],
            sentiments_pd[sentiments_pd.Account=='fox']['Compound'],
           label='FOX', color='blue', edgecolors='black')
plt.scatter(sentiments_pd[sentiments_pd.Account=='CBS']['Tweets Ago'],
            sentiments_pd[sentiments_pd.Account=='CBS']['Compound'],
           label='CBS', color='green', edgecolors='black')

ax=plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])
plt.ylim(-1, 1, 0.5)

plt.grid(True)
plt.grid(color="grey")
plt.title("Sentiment Analysis of Media Tweets (%s)" % (time.strftime("%x")), fontsize=20)
plt.xlabel('Tweets Ago')
plt.ylabel('Tweet Polarity')

#Create legend from custom artist/label lists
type1 = plt.scatter([],[], s=75, marker='o', color='cyan', edgecolors='black')
type2 = plt.scatter([],[], s=75, marker='o', color='green', edgecolors='black')
type3 = plt.scatter([],[], s=75, marker='o', color='red', edgecolors='black')
type4 = plt.scatter([],[], s=75, marker='o', color='blue', edgecolors='black')
type5 = plt.scatter([],[], s=75, marker='o', color='yellow', edgecolors='black')
sns.set()

plt.legend((type1,type2,type3,type4,type5),
           ('BBC', 'CBS', 'CNN', 'Fox', 'New York Times'),
            scatterpoints=1,
            loc=2, 
            borderaxespad=0.,
            bbox_to_anchor=(1, 1),
            fontsize=10,
            title='Media Sources')
plt.savefig("Sentiment_Analysis_Media.png")
plt.show()
```


![png](output_7_0.png)



```python
sentiment = sentiments_pd.groupby(["Account"], as_index = False)['Compound'].mean()
sentiment
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Account</th>
      <th>Compound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BBC</td>
      <td>0.128065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CBS</td>
      <td>0.378797</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CNN</td>
      <td>0.078330</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fox</td>
      <td>0.239635</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nytimes</td>
      <td>-0.027433</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(15,10))
sns.set()
colors =["cyan", "green","red", "blue", "yellow"]
plt.bar(sentiment['Account'],sentiment['Compound'],color=colors,alpha=0.5,align='center')
plt.ylim(-0.4, 0.4, 0.5)
plt.title("Overall Media Sentiment Based on Twitter (%s)" % (time.strftime("%x")), fontsize=20)
plt.ylabel("Tweets Polarity")
plt.savefig("Overall_Media_Sentiment.png")
plt.show()
```


![png](output_9_0.png)

