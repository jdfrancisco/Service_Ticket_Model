import serviceTicket_config
import torch
import transformers
import torch.nn as nn

def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss

class EntityModel(nn.Module):
    def __init__(self, num_topic):
        super(EntityModel, self).__init__()
        self.num_topic = num_topic
        self.bert = transformers.BertModel.from_pretrained(serviceTicket_config.BASE_MODEL_PATH, return_dict=False)
        self.bert_drop_1 = nn.Dropout(0.3)
        self.out_topic = nn.Linear(768, self.num_topic)

    def forward(self, ids, mask, token_type_ids, target_topic):
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

        bo_topic = self.bert_drop_1(o1)

        topic = self.out_topic(bo_topic)

        loss_topic = loss_fn(topic, target_topic, mask, self.num_topic)

        loss = (loss_topic) / 1 # number of variables

        return topic, loss