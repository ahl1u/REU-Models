import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

# Read the data from the file
df = pd.read_csv('sentiment_classification_result_roBERTa.csv')

# Map 'vape_user' to 'vapers' and 'non-vapers'
df['vape_user'] = df['vape_user'].map({'y': 'vapers', 'n': 'non-vapers'})

# Map 'sentiment_score' to 'Negative', 'Neutral', 'Positive'
df['sentiment_score'] = df['sentiment_score'].map({-1: 'Negative', 0: 'Neutral', 1: 'Positive'})

# Group by 'vape_user' and 'sentiment_score' and count the number of posts
counts_df = df.groupby(['vape_user', 'sentiment_score']).size().unstack(fill_value=0)

# Check the resulting counts DataFrame
print("Counts of each sentiment category for vapers and non-vapers:")
print(counts_df)

# Perform two-proportion Z-tests for each sentiment category
sentiment_categories = ['Negative', 'Neutral', 'Positive']

for sentiment in sentiment_categories:
    # Get counts of the chosen sentiment and total counts for each group
    count = counts_df.loc[:, sentiment].values
    nobs = counts_df.sum(axis=1).values

    # Check the count and nobs arrays
    print(f"\nCounts of '{sentiment}' sentiment: {count}")
    print(f"Total counts: {nobs}")

    # Conduct the two-proportion Z-test
    z, p = proportions_ztest(count, nobs)

    print(f"\nTwo-proportion Z-test for sentiment category '{sentiment}':")
    print(f"Z-statistic = {z:.3f}")
    print(f"p-value = {p:.3e}")
    print('---')
