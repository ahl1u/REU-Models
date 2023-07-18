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

def main():
    # Step 2: Load and preprocess the data
    data = pd.read_csv('labeled_tweets.csv')

    # Convert 'vape_user' field to binary format
    data['vape_user'] = data['vape_user'].map({'y': 1, 'n': 0})
    data['sentiment'] = data['sentiment'].map({-1: 0, 0: 1, 1: 2})

    # Step 3: Instantiate the tokenizer and set max_len
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    max_len = 256

    # Step 4: Prepare K-fold Cross validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    inputs = data.text.to_numpy()
    labels = data.sentiment.to_numpy()
    vape_users = data.vape_user.to_numpy()

    print(len(inputs))
    print(len(labels))
    print(len(vape_users))

    f1_scores = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(inputs, labels)):
        print(f'Fold: {fold}')

        # Print sizes and ranges for debugging
        print(f'Size of labels array: {len(labels)}')
        print(f'Size of train_ids array: {len(train_ids)}, min value: {min(train_ids)}, max value: {max(train_ids)}')
        print(f'Size of val_ids array: {len(val_ids)}, min value: {min(val_ids)}, max value: {max(val_ids)}')

        train_inputs, val_inputs = inputs[train_ids], inputs[val_ids]
        print(f'Size of train_inputs: {len(train_inputs)}')  # should be 1323
        print(f'Size of val_inputs: {len(val_inputs)}')  # should be 147

        train_labels, val_labels = labels[train_ids], labels[val_ids]
        print(f'Size of train_labels: {len(train_labels)}')  # should be 1323
        print(f'Size of val_labels: {len(val_labels)}')  # should be 147
        
        train_vape_users, val_vape_users = vape_users[train_ids], vape_users[val_ids]

        train_dataset = SentimentAnalysisDataset(
            tweets=train_inputs,
            labels=train_labels,
            vape_users=train_vape_users,
            tokenizer=tokenizer,
            max_len=max_len
        )
        val_dataset = SentimentAnalysisDataset(
            tweets=val_inputs,
            labels=val_labels,
            vape_users=val_vape_users,
            tokenizer=tokenizer,
            max_len=max_len
        )

        # Step 5: Load the model
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

        optim = torch.optim.AdamW(model.parameters(), lr=1e-5)

        # Define checkpoint file for current fold
        CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, f'checkpoint_{fold}.pt')

        # Load checkpoint if it exists
        start_epoch = 0
        if os.path.isfile(CHECKPOINT_FILE):
            checkpoint = torch.load(CHECKPOINT_FILE)
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

            print('Epoch:', epoch, 'loss:', running_loss)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss,
            }, CHECKPOINT_FILE)

        # Step 7: Evaluate the model
        model.eval()

        val_data_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)
        total_correct = 0
        total_count = 0

        # Define lists to store all true labels and predictions
        all_true_labels = []
        all_predicted_labels = []

        for data in tqdm(val_data_loader):
            inputs = data['input_ids'].to(device)
            masks = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            with torch.no_grad():
                outputs = model(inputs, attention_mask=masks)

            predictions = torch.argmax(outputs.logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_count += labels.numel()

            # Append current batch of true labels and predictions to respective lists
            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predictions.cpu().numpy())

        print('Test accuracy:', total_correct / total_count)

        # After the loop, convert the lists to numpy arrays
        all_true_labels = np.array(all_true_labels)
        all_predicted_labels = np.array(all_predicted_labels)

        # Now, you can calculate the F1 score
        f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')
        f1_scores.append(f1)

        print('Validation F1 Score:', f1)

    print('Average F1 Score:', np.mean(f1_scores))

    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)

class SentimentAnalysisDataset(Dataset):
    def __init__(self, tweets, labels, vape_users, tokenizer, max_len):
        self.tweets = tweets
        self.labels = labels
        self.vape_users = vape_users
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        label = self.labels[item]
        vape_user = self.vape_users[item]

        try:
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
        except Exception as e:
            print(f"Error with encoding tweet at index {item}")
            raise e

        try:
            return {
                'tweet_text': tweet,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(int(label), dtype=torch.long),
                'vape_user': torch.tensor(vape_user, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error with converting to tensor at index {item}")
            raise e

if __name__ == '__main__':
    main()
