import pandas as pd
import numpy as np

import joblib
import torch
import csv

import serviceTicket_config
import serviceTicket_dataset
import serviceTicket_engine
from serviceTicket_model import EntityModel

if __name__ == "__main__":

    meta_data = joblib.load("meta.bin")
    enc_topic = meta_data["enc_topic"]

    num_topic = len(list(enc_topic.classes_))

    complaints = input("Enter a complaint: ")
    tokenized_complaints = serviceTicket_config.TOKENIZER.encode(complaints)

    complaints = complaints.split()
    print(complaints)
    print(tokenized_complaints)

    test_dataset = serviceTicket_dataset.EntityDataset(
        complaints=[complaints], 
        topics=[[0] * len(complaints)],
    )

    device = torch.device("cpu")
    model = EntityModel(num_topic=num_topic)
    model.load_state_dict(torch.load(serviceTicket_config.MODEL_PATH))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        topic, _ = model(**data)

        print(enc_topic.inverse_transform(topic.argmax(2).cpu().numpy().reshape(-1))[:len(tokenized_complaints)])
    
    extracted_data = pd.read_csv('../output/Extracted_Data.csv')
    last_ticket = extracted_data["Ticket #"].iloc[-1]

    # List that we want to add as a new row
    append_row = [last_ticket + 1, complaints, enc_topic.inverse_transform(topic.argmax(2).cpu().numpy().reshape(-1))[:len(tokenized_complaints)]]
    
    # Open our existing CSV file in append mode
    # Create a file object for this file
    with open('../output/Extracted_Data.csv', 'a') as f_object:
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = csv.writer(f_object)
    
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(append_row)
    
        # Close the file object
        f_object.close()

