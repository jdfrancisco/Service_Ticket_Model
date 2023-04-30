import serviceTicket_config
import torch

class EntityDataset:
    def __init__(self, complaints, topics):
        # complaints: [["Issue is..."], [....].....]]
        # topics: [[account services], [....].....]]
        self.complaints = complaints
        self.topics = topics
    
    def __len__(self):
        return len(self.complaints)
    
    def __getitem__(self, item):
        complaint = self.complaints[item]
        topic = self.topics[item]

        ids = []
        target_topic =[]

        for i, s in enumerate(complaint):
            inputs = serviceTicket_config.TOKENIZER.encode(
                str(s),
                add_special_tokens=False
            )
            input_len = len(inputs)
            ids.extend(inputs)
            target_topic.extend([topic[i]] * input_len)

        ids = ids[:serviceTicket_config.MAX_LEN - 2]
        target_topic = target_topic[:serviceTicket_config.MAX_LEN - 2]

        ids = [101] + ids + [102]
        target_topic = [0] + target_topic + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = serviceTicket_config.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_topic = target_topic + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_topic": torch.tensor(target_topic, dtype=torch.long),  
        }