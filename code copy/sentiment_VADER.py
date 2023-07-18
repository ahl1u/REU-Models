import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

analyzer = SentimentIntensityAnalyzer()


with open('pre_sentiment.csv', 'r') as sentiment_file: #input file
    reader = csv.reader(sentiment_file)
    next(reader)
    with open('sentiment_result_VADER.csv', 'w') as output_file:   #output file
        writer = csv.writer(output_file)
        writer.writerow(["posts","keywords","sentiment_score"]) #column 1 is the Twitter post, column 2 is the flavor name, column 3 is the sentiment score
        for line in reader:
            post = re.sub(r'\s+', ' ', line[2])
            sentiment_score = analyzer.polarity_scores(post)['compound']  #sentiment analysis
            writer.writerow([line[0],line[1], line[2], sentiment_score])   #write output file