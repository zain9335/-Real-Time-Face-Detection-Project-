import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import re

# Read 'chat.txt' file line by line
with open('chat.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
print(lines)
# Remove metadata and keep only the message
lines = [re.sub(r'\[.*\]', '', line).strip() for line in lines if line.strip()]
# print(lines)

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Perform sentiment analysis on each message
sentiment_scores = [sia.polarity_scores(line) for line in lines]
print(sentiment_scores)
compound_scores = [score['compound'] for score in sentiment_scores]
positive_scores = [score['pos'] for score in sentiment_scores]
negative_scores = [score['neg'] for score in sentiment_scores]
neutral_scores = [score['neu'] for score in sentiment_scores]

# Plot sentiment scores over the message index
plt.plot(range(len(lines)), compound_scores, label='Compound')
plt.plot(range(len(lines)), positive_scores, label='Positive')
plt.plot(range(len(lines)), negative_scores, label='Negative')
plt.plot(range(len(lines)), neutral_scores, label='Neutral')

# Set plot labels and legend
plt.xlabel('Message Index')
plt.ylabel('Sentiment Score')
plt.legend()

# Show the plot
plt.show()
