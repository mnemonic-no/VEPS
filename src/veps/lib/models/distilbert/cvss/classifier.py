import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification

class MultiOutputDistilBert(DistilBertForSequenceClassification):
    def __init__(self, config, num_labels_list):
        super().__init__(config)
        self.pre_classifier = None
        self.classifier = None
        
        self.classifiers = nn.ModuleList(
            [nn.Linear(config.dim, num_labels) for num_labels in num_labels_list]
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0]
        logits = [classifier(pooled_output) for classifier in self.classifiers]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = sum(loss_fct(logit, label) for logit, label in zip(logits, labels))
            return {"loss": loss, "logits": logits} if return_dict else (loss, logits)

        return {"logits": logits} if return_dict else logits