from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from bertopic import BERTopic
import plotly.offline as pyo
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load your data from a CSV file
df = pd.read_csv('reduced_tweets_positive.csv') 

# Download necessary nltk resources
nltk.download('stopwords')
nltk.download('wordnet')

# Create lemmatizer
lemmatizer = WordNetLemmatizer()

# Define stopwords
stop_words = set(stopwords.words('english'))

# Preprocess the text data
df['processed_text'] = df['text'].str.lower()  # Lowercase
df['processed_text'] = df['processed_text'].str.replace('[^\w\s]', '')  # Remove punctuation
df['processed_text'] = df['processed_text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split() if word not in stop_words]))  # Lemmatize and remove stopwords

# Convert processed text to list
documents = df['processed_text'].tolist()

# Create a BERTopic model
topic_model = BERTopic(language="english", calculate_probabilities=True)

# Fit the model with the documents
topics, probabilities = topic_model.fit_transform(documents)

# Reduce the topics
topic_model.reduce_topics(docs=documents, nr_topics=10)

# Use the reduced model to predict topics for the same documents
new_topics, new_probs = topic_model.transform(documents)

# Update the dataframe with the new topics and probabilities
df['topic'] = new_topics
df['probability'] = [max(prob) if len(prob) > 0 else 0 for prob in new_probs]

# Visualize the topics and get the figure
fig = topic_model.visualize_topics()

# Display the figure in a web browser
pyo.plot(fig)

# Get the unique reduced topics from the new_topics
unique_topics = pd.Series(new_topics).unique()

# Extract each unique topic
for unique_topic in unique_topics:
    topic = topic_model.get_topic(unique_topic)
    # Check if the topic is found
    if topic is not False:
        # Save individual topic to a csv file
        with open(f'BERTopic_topicP_{unique_topic}.csv', 'w') as f:
            for word, score in topic:
                f.write(f"{word},{score}\n")

# Prepare a list to store each topic's details
topics_data = []

for unique_topic in unique_topics:
    topic_repr = topic_model.get_topic(unique_topic)
    
    # Make sure the topic is found
    if topic_repr is not False:
        # Extract individual topic information
        topic_name = "_".join([word for word, _ in topic_repr[:5]])
        topic_words = [word for word, _ in topic_repr]

        # Get representative documents for this topic
        representative_docs = df[df['topic'] == unique_topic]['text'].values.tolist()[:3]  # Get first 3 documents

        # Create a dictionary to store this topic's information
        topic_info = {
            "Topic": unique_topic,
            "Count": len(df[df['topic'] == unique_topic]),
            "Name": f"{unique_topic}_{topic_name}",
            "Representation": str(topic_words),
            "Representative_Docs": str(representative_docs)
        }

        # Append the dictionary to the topics_data list
        topics_data.append(topic_info)


# Convert the list of dictionaries into a DataFrame
topics_df = pd.DataFrame(topics_data)

# Save the DataFrame to a new CSV file
topics_df.to_csv('reduced_topics_informationP.csv', index=False)

# Save the DataFrame to a new CSV file
df.to_csv('tweets_with_topicsP.csv', index=False)
