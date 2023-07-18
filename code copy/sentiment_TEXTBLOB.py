from textblob import TextBlob
import csv
from datetime import datetime
from collections import defaultdict

def preprocess_text(text):
    replacements = {
        "idc": "I don't care",
        "lol": "laugh out loud",
        "btw": "by the way",
        "omg": "oh my god",
        "brb": "be right back",
        "jk": "just kidding",
        "tbh": "to be honest",
        "afaik": "as far as I know",
        "imo": "in my opinion",
        "fomo": "fear of missing out",
        "imo": "in my opinion",
        "hmu": "hit me up",
        "omw": "on my way",
        "wtf": "what the f***",
        "rn": "right now",
        "tbt": "throwback Thursday",
        "bff": "best friends forever",
        "fwiw": "for what it's worth",
        "ikr": "I know, right?",
        "icymi": "in case you missed it",
        "smh": "shake my head",
        "nvm": "nevermind",
        "tmi": "too much information",
        "imho": "in my humble opinion",
        "fyi": "for your information",
        "gtg": "got to go",
        "np": "no problem",
        "btw": "by the way",
        "yw": "you're welcome",
        "gg": "good game",
        "gr8": "great",
        "thx": "thanks",
        "omg": "oh my god",
        "jk": "just kidding",
        "np": "no problem",
        "pls": "please",
        "rly": "really",
        "u": "you",
        "ur": "your",
        "yr": "year",
        "tho": "though",
        "thru": "through",
        "b4": "before",
        "cuz": "because",
        "wanna": "want to",
        "gonna": "going to",
        "gotta": "got to",
        "kinda": "kind of",
        "sorta": "sort of",
        "lemme": "let me",
        "gimme": "give me",
        "ain't": "am not / is not / are not",
        "gon": "going to",
        "ya": "you",
        "nbd": "no big deal",
        "fam": "family",
        "sry": "sorry",
        "probs": "probably",
        "thx": "thanks",
        "thnx": "thanks",
        "yo": "hey",
        "dope": "excellent",
        "lit": "awesome",
        "sick": "cool",
        "legit": "legitimate",
        "bruh": "bro",
        "hella": "very",
        "g2g": "got to go",
        "omw": "on my way",
        "bday": "birthday",
        "af": "as f***",
        "tgif": "thank goodness it's Friday",
        "meh": "indifferent",
        "fomo": "fear of missing out",
        "hmu": "hit me up",
        "ik": "I know",
        "idk": "I don't know",
        "irl": "in real life",
        "ttyl": "talk to you later",
        "wb": "welcome back",
        "yolo": "you only live once",
        "bff": "best friends forever",
        "fb": "Facebook",
        "ig": "Instagram",
        "iggy": "I guess",
        "insta": "Instagram",
        "lmao": "laughing my ass off",
        "lmfao": "laughing my f***ing ass off",
        "rofl": "rolling on the floor laughing",
        "smh": "shake my head",
        "stfu": "shut the f*** up",
        "tmi": "too much information",
        "ttyl": "talk to you later",
        "wth": "what the hell",
        "omg": "oh my god",
        "wtf": "what the f***",
        "gtfo": "get the f*** out",
        "jk": "just kidding",
        "jk lol": "just kidding, laugh out loud",
        "np": "no problem",
        "npnp": "no problem, no problem",
        "pls": "please",
        "plz": "please",
        "sry": "sorry",
        "thx": "thanks",
        "thnx": "thanks",
        "ty": "thank you",
        "tyvm": "thank you very much",
        "yw": "you're welcome"
        # add other replacements here
    }
    
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

with open('pre_sentiment.csv', 'r') as sentiment_file:  # input file
    reader = csv.reader(sentiment_file)
    next(reader)
    
    with open('sentiment_result_textblob.csv', 'w') as output_file:  # output file
        writer = csv.writer(output_file)
        writer.writerow(["user_id", "timestamp", "posts", "keywords", "sentiment_score"])  # column headers

        biweek_sentiment = defaultdict(list)
        
        for line in reader:
            post = preprocess_text(line[2])
            sentiment = TextBlob(post).sentiment.polarity
            
            # Parse the timestamp and get the bi-week number
            timestamp = datetime.strptime(line[1], "%a %b %d %H:%M:%S +0000 %Y")
            biweek_number = f'{timestamp.year}-{timestamp.isocalendar()[1] // 2}'  # calculate bi-week number
            
            # Append the sentiment to the list for this bi-week
            biweek_sentiment[biweek_number].append(sentiment)
            
            writer.writerow([line[0], line[1], line[2], sentiment])  # write output file

# Now write the average sentiment per bi-week to another file
with open('average_sentiment_per_biweek.csv', 'w') as avg_file:
    writer = csv.writer(avg_file)
    writer.writerow(["biweek", "average_sentiment"])  # column headers
    
    for biweek, sentiments in biweek_sentiment.items():
        avg_sentiment = sum(sentiments) / len(sentiments)
        writer.writerow([biweek, avg_sentiment])
