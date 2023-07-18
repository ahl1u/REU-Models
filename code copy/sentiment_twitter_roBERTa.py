import csv
from transformers import pipeline
import re
import preprocessor as p  # Twitter specific preprocessing library

sentiment_classifier = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment')

def get_sentiments(texts):
    results = sentiment_classifier(texts)
    return results

p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY)  # Clean URLs, Emojis, and Smileys

with open('pre_sentiment.csv', 'r') as sentiment_file:  # input file
    reader = csv.reader(sentiment_file)
    next(reader)
    
    with open('sentiment_result_twitter.csv', 'w') as output_file:  # output file
        writer = csv.writer(output_file)
        writer.writerow(["user_id", "timestamp", "posts", "keywords", "sentiment_score"])  # column headers

        batch = []
        batch_lines = []
        for line in reader:
            post = p.clean(line[2])[:512]  # Clean and limit to 512 tokens
            batch.append(post)
            batch_lines.append(line)

            # Process in batches of size 100
            if len(batch) == 100:
                sentiment_results = get_sentiments(batch)
                for sentiment_result, batch_line in zip(sentiment_results, batch_lines):
                    sentiment_score = sentiment_result['score'] if sentiment_result['label'] == 'POSITIVE' else -sentiment_result['score']
                    writer.writerow([batch_line[0], batch_line[1], batch_line[2], sentiment_score])  # write output file

                # Clear the batch
                batch = []
                batch_lines = []

        # Process the last batch
        if batch:
            sentiment_results = get_sentiments(batch)
            for sentiment_result, batch_line in zip(sentiment_results, batch_lines):
                sentiment_score = sentiment_result['score'] if sentiment_result['label'] == 'POSITIVE' else -sentiment_result['score']
                writer.writerow([batch_line[0], batch_line[1], batch_line[2], sentiment_score])  # write output file
