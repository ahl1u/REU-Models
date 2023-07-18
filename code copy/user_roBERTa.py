from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import csv
import re

model_path = "./binary_classification_model"  # updated directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

classification = pipeline('text-classification', model=model, tokenizer=tokenizer)

def get_vape_user(texts):
    results = classification(texts)
    # Convert to 'y' or 'n'
    for result in results:
        if result['label'] == "LABEL_0":
            result['score'] = 'n'
        elif result['label'] == "LABEL_1":
            result['score'] = 'y'
    return results

with open('sentiment_result_roBERTa.csv', 'r') as sentiment_file:  # input file
    reader = csv.reader(sentiment_file)
    next(reader)
    
    with open('sentiment_classification_result_roBERTa.csv', 'w') as output_file:  # output file
        writer = csv.writer(output_file)
        writer.writerow(["user_id", "timestamp", "posts", "keywords", "sentiment_score", "vape_user"])  # column headers

        batch = []
        batch_lines = []
        for line in reader:
            post = re.sub(r'\s+', ' ', line[2])[:512]  # BERT has a maximum input length of 512 tokens
            batch.append(post)
            batch_lines.append(line)

            # Process in batches of size 100
            if len(batch) == 100:
                vape_user_results = get_vape_user(batch)
                for vape_user_result, batch_line in zip(vape_user_results, batch_lines):
                    vape_user = vape_user_result['score']  # Directly write the classification result
                    writer.writerow([batch_line[0], batch_line[1], batch_line[2], batch_line[3], vape_user])  # write output file

                # Clear the batch
                batch = []
                batch_lines = []

        # Process the last batch
        if batch:
            vape_user_results = get_vape_user(batch)
            for vape_user_result, batch_line in zip(vape_user_results, batch_lines):
                    vape_user = vape_user_result['score']  # Directly write the classification result
                    writer.writerow([batch_line[0], batch_line[1], batch_line[2], batch_line[3], vape_user])  # write output file
