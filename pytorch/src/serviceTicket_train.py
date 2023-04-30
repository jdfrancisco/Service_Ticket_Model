import pandas as pd
import numpy as np

import joblib
import torch
import csv

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import serviceTicket_config
import serviceTicket_dataset
import serviceTicket_engine
from serviceTicket_model import EntityModel


def process_data(data_path):
    df = pd.read_csv(data_path, nrows=1000, encoding="latin-1")
    df.loc[:, "Complaint #"] = df["Complaint #"].fillna(method="ffill")

    enc_topic = preprocessing.LabelEncoder()

    df.loc[:, "Topic"] = enc_topic.fit_transform(df["Topic"])

    complaints = df.groupby("Complaint #")["Complaint_clean"].apply(list).values
    topic = df.groupby("Complaint #")["Topic"].apply(list).values
    return complaints, topic, enc_topic

if __name__ == "__main__":
    complaints, topic, enc_topic= process_data(serviceTicket_config.TRAINING_FILE)
    
    meta_data = {
        "enc_topic": enc_topic
    }

    joblib.dump(meta_data, "meta.bin")

    num_topic = len(list(enc_topic.classes_))

    (
        train_complaints,
        test_complaints,
        train_topic,
        test_topic
    ) = model_selection.train_test_split(complaints, topic, random_state=42, test_size=0.1)

    train_dataset = serviceTicket_dataset.EntityDataset(
        complaints=train_complaints, topics=train_topic
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=serviceTicket_config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = serviceTicket_dataset.EntityDataset(
        complaints=test_complaints, topics=test_topic
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=serviceTicket_config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cpu")
    model = EntityModel(num_topic=num_topic)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_complaints) / serviceTicket_config.TRAIN_BATCH_SIZE * serviceTicket_config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    with open('../output/Loss_estimates.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["EPOCH", "Train_Loss", "Valid_Loss"])
        best_loss = np.inf
        for epoch in range(serviceTicket_config.EPOCHS):
            train_loss = serviceTicket_engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
            test_loss = serviceTicket_engine.eval_fn(valid_data_loader, model, device)
            print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
            writer.writerow([epoch, train_loss, test_loss])
            if test_loss < best_loss:
                torch.save(model.state_dict(), serviceTicket_config.MODEL_PATH)
                best_loss = test_loss