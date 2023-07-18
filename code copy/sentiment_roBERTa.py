from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import csv
import re

model_path = "./"  # current directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

sentiment_classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

def get_sentiments(texts):
    results = sentiment_classifier(texts)
    # Convert to -1, 0, 1
    for result in results:
        if result['label'] == "LABEL_0":
            result['score'] = -1
        elif result['label'] == "LABEL_1":
            result['score'] = 0
        elif result['label'] == "LABEL_2":
            result['score'] = 1
    return results

with open('pre_sentiment.csv', 'r') as sentiment_file:  # input file
    reader = csv.reader(sentiment_file)
    next(reader)
    
    with open('sentiment_result_roBERTa.csv', 'w') as output_file:  # output file
        writer = csv.writer(output_file)
#        writer.writerow(["user_id", "timestamp", "posts", "keywords", "sentiment_score"])  # column headers
        writer.writerow(["text", "sentiment_score"])  # column headers

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
                    sentiment_score = sentiment_result['score']  # Directly write the sentiment score
                    writer.writerow([batch_line[0], batch_line[1], batch_line[2], sentiment_score])  # write output file

                # Clear the batch
                batch = []
                batch_lines = []

        # Process the last batch
        if batch:
            sentiment_results = get_sentiments(batch)
            for sentiment_result, batch_line in zip(sentiment_results, batch_lines):
                    sentiment_score = sentiment_result['score']  # Directly write the sentiment score
                    writer.writerow([batch_line[0], batch_line[1], batch_line[2], sentiment_score])  # write output file
