import csv
from transformers import pipeline
import re

sentiment_classifier = pipeline('sentiment-analysis')

def get_sentiments(texts):
    results = sentiment_classifier(texts)
    return results

with open('pre_sentiment.csv', 'r') as sentiment_file:  # input file
    reader = csv.reader(sentiment_file)
    next(reader)
    
    with open('sentiment_result2.csv', 'w') as output_file:  # output file
        writer = csv.writer(output_file)
        writer.writerow(["user_id", "timestamp", "posts", "keywords", "sentiment_score"])  # column headers

        batch = []
        batch_lines = []
        for line in reader:
            post = re.sub(r'\s+', ' ', line[2])[:512]  # BERT has a maximum input length of 512 tokens
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