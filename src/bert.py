import os
import numpy as np

import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

BERT_MODEL_NAME = "bert-base-uncased"
BERT_N_CLASSES = 6
BERT_MAX_LENGTH = 128
BERT_BATCH_SIZE = 16
BERT_EPOCHS = 5
BERT_LEARNING_RATE = 2e-5
BERT_PATH = "../models/bert_classifier.pth"
BERT_TRAIN_PATH = "../dataset/train.csv"
BERT_TEST_PATH = "../dataset/test.csv"
BERT_TEST_LABELS_PATH = "../dataset/test_labels.csv"


class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = BERT_MAX_LENGTH

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label),
        }


class BERTClassifier(torch.nn.Module):
    def __init__(self, device: str, load_from_path: bool = False):
        super(BERTClassifier, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME).to(device)
        self.classifier = torch.nn.Linear(
            self.bert.config.hidden_size, BERT_N_CLASSES
        ).to(device)

        if load_from_path and os.path.exists(BERT_PATH):
            self.bert.load_state_dict(
                torch.load(
                    f=BERT_PATH,
                    map_location=torch.device(device),
                )
            )

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.classifier(bert_output.pooler_output)

        return output

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=BERT_BATCH_SIZE, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=BERT_BATCH_SIZE)

    def setup(self):
        df_train = pd.read_csv(BERT_TRAIN_PATH)
        df_val = pd.merge(
            pd.read_csv(BERT_TEST_PATH),
            pd.read_csv(BERT_TEST_LABELS_PATH),
            on="id",
            how="inner",
        )
        self.classes = list(
            filter(lambda x: x not in ["id", "comment_text"], df_train.columns)
        )

        self.train_dataset = BERTDataset(
            df_train["comment_text"].values.tolist(),
            df_train[self.classes].values.tolist(),
            self.tokenizer,
        )
        self.val_dataset = BERTDataset(
            df_val["comment_text"].values.tolist(),
            df_val[self.classes].values.tolist(),
            self.tokenizer,
        )

    def evaluate(self, data_loader, device):
        total_loss = 0
        predictions = []
        actual_labels = []
        self.bert.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
                input_ids = batch["input_ids"].to(device, dtype=torch.long)
                attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
                labels = batch["label"].to(device, dtype=torch.float)
                outputs = self.forward(input_ids, attention_mask)
                loss = torch.nn.BCEWithLogitsLoss()(outputs, labels)
                total_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(data_loader)
        avg_accuracy = np.sum(predictions == actual_labels) / len(data_loader)
        report = classification_report(actual_labels, predictions)

        return avg_loss, avg_accuracy, report

    def train(self, data_loader, optimizer, scheduler, device):
        total_loss = 0
        predictions = []
        actual_labels = []
        self.bert.train()
        for batch in tqdm(data_loader, desc="Training", unit="batch"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device, dtype=torch.long)
            attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
            labels = batch["label"].to(device, dtype=torch.float)
            outputs = self.forward(input_ids, attention_mask)
            loss = torch.nn.BCEWithLogitsLoss()(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
        
        avg_loss = total_loss / len(data_loader)
        avg_accuracy = np.sum(predictions == actual_labels) / len(data_loader)

        return avg_loss, avg_accuracy

    def prediction(self, input, device, max_length=128):
        self.bert.eval()
        encoding = self.tokenizer(
            input,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        input_ids = encoding["input_ids"].to(device, dtype=torch.long)
        attention_mask = encoding["attention_mask"].to(device, dtype=torch.long)

        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            return preds


def train_bert(device: str, model: BERTClassifier):
    logger.info("> Preparing datasets...")
    model.setup()
    train_dataloader = model.train_dataloader()
    val_dataloader = model.val_dataloader()

    logger.info("> Loading optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=BERT_LEARNING_RATE)
    total_steps = len(train_dataloader) * BERT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    logger.info("> Train/Evaluate...")
    for epoch in range(BERT_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{BERT_EPOCHS}")
        train_loss, train_acc = model.train(
            train_dataloader, optimizer, scheduler, device
        )
        val_loss, val_acc, report = model.evaluate(val_dataloader, device)
        logger.info(
            "  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"
            % (train_loss, val_loss, train_acc, val_acc)
        )

    logger.info(f"\n{report}")
    logger.info("> Saving model")
    torch.save(model.bert.state_dict(), BERT_PATH)
    logger.info("> Done!")


if __name__ == "__main__":
    model = BERTClassifier("cuda")
    train_bert("cuda", model)
