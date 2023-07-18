# Step 1: Import necessary libraries
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, AdamW
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

CHECKPOINT_DIR = './'  # Choose your directory to save checkpoints
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'binary_classification_checkpoint_{fold}.pt')

def main():
    # Step 2: Load and preprocess the data
    data = pd.read_csv('labeled_tweets.csv')

    # Check for NaN values in 'vape_user' column and drop rows containing them
    if data['vape_user'].isna().sum() > 0:
        print("Number of NaN values before dropping:", data['vape_user'].isna().sum())
        data = data.dropna(subset=['vape_user'])
        print("Number of NaN values after dropping:", data['vape_user'].isna().sum())

    print("Unique values in 'vape_user' before mapping:", data['vape_user'].unique())

    # Convert 'vape_user' field to binary format
    data['vape_user'] = data['vape_user'].map({'y': 1, 'n': 0})

    # Step 3: Instantiate the tokenizer and set max_len
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    max_len = 256

    # Step 4: Prepare K-fold Cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inputs = data.text.to_numpy()
    labels = data.vape_user.to_numpy()

    f1_scores = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(inputs, labels)):
        print(f'Fold: {fold}')

        train_inputs, val_inputs = inputs[train_ids], inputs[val_ids]
        train_labels, val_labels = labels[train_ids], labels[val_ids]

        train_dataset = SentimentAnalysisDataset(
            tweets=train_inputs,
            labels=train_labels,
            tokenizer=tokenizer,
            max_len=max_len
        )
        val_dataset = SentimentAnalysisDataset(
            tweets=val_inputs,
            labels=val_labels,
            tokenizer=tokenizer,
            max_len=max_len
        )

        # Step 5: Load the model
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

        optim = torch.optim.AdamW(model.parameters(), lr=1e-5)

        # Load checkpoint if it exists
        start_epoch = 0
        if os.path.isfile(CHECKPOINT_FILE.format(fold=fold)):
            checkpoint = torch.load(CHECKPOINT_FILE.format(fold=fold))
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'Loaded checkpoint from epoch {start_epoch}')

        # Step 6: Train the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        data_loader = DataLoader(train_dataset, batch_size=32, num_workers=4)

        for epoch in range(start_epoch, start_epoch + 3):  # loop over the dataset multiple times
            running_loss = 0.0
            model.train()  # Switch to train mode
            for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data['input_ids'].to(device)
                masks = data['attention_mask'].to(device)
                labels = data['labels'].to(device)

                # zero the parameter gradients
                optim.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs, attention_mask=masks, labels=labels)
                loss = outputs[0]
                loss.backward()
                optim.step()

                # print statistics
                running_loss += loss.item()

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': loss,
                }, CHECKPOINT_FILE.format(fold=fold))

            print('Epoch:', epoch, 'loss:', running_loss)

            # Validation and F1 Score Calculation
            val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)
            val_predictions, val_true_labels = [], []

            model.eval()  # Switch to eval mode
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch['input_ids'].to(device)
                    masks = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(inputs, attention_mask=masks)
                    _, predicted = torch.max(outputs.logits, 1)

                    val_predictions.extend(predicted.cpu().numpy())
                    val_true_labels.extend(labels.cpu().numpy())

            # Calculate F1 Score
            f1 = f1_score(val_true_labels, val_predictions, average='weighted')
            f1_scores.append(f1)
            print(f'Validation F1 Score (epoch {epoch}): {f1}')

        print(f'Average F1 Score: {np.mean(f1_scores)}')

    model.save_pretrained('./binary_classification_model/')
    tokenizer.save_pretrained('./binary_classification_model/')



class SentimentAnalysisDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer, max_len):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        label = self.labels[item]

        if pd.isna(label):
            print(f"Found NaN label at index {item}")

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True  # Explicit truncation
        )

        try:
            return {
                'tweet_text': tweet,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(int(label), dtype=torch.long)
            }
        except RuntimeError as e:
            print(f"Error with label {label} at index {item}")
            raise e

if __name__ == '__main__':
    main()
