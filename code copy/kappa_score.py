import pandas as pd
from sklearn.metrics import cohen_kappa_score

def compare_files(file1, file2):
    # Load data
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Check if text columns are identical
    if all(df1['text'] == df2['text']):
        # Calculate and print Cohen's Kappa for sentiment and vape_user columns
        sentiment_kappa = cohen_kappa_score(df1['sentiment'], df2['sentiment'])
        vape_user_kappa = cohen_kappa_score(df1['vape_user'], df2['vape_user'])

        print(f"Cohen's Kappa for sentiment: {sentiment_kappa}")
        print(f"Cohen's Kappa for vape_user: {vape_user_kappa}")
    else:
        print("Text columns are not identical")

# Call the function
compare_files('labeled_tweets_AGAIN.csv', 'labeled_tweets_AGAIN_labeled.csv')
