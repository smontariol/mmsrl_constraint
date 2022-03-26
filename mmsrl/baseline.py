import torch
import transformers

import mmsrl.loss


class Model(torch.nn.Module):
    def __init__(self, config, label_weights):
        super().__init__()
        self.config = config
        self.transformer = transformers.AutoModel.from_pretrained(self.config.modelname)
        # TODO tune the smaller output size for the linears to compute attention?
        self.text_linear= torch.nn.Linear(self.transformer.config.hidden_size, self.config.hidden_size)
        self.entities_linear= torch.nn.Linear(self.transformer.config.hidden_size, self.config.hidden_size)
        self.classifier = torch.nn.Linear(self.transformer.config.hidden_size, 4) # output * number of classes
        #self.relu = torch.nn.ReLU()
        self.loss = mmsrl.loss.Loss(self.config, label_weights)

    def forward(self, input_ids, attention_mask, entities_mask, entities_input_ids, entities_attention_mask, labels=None, **kwargs):
        # entities are tri-dimensionnal (batch * nb of entities * emb dim) while text is bi-dimensional
        nb_entities = entities_input_ids.shape[1]
        batch_size = entities_input_ids.shape[0]
        fake_batch_size = batch_size*nb_entities

        text_output = self.transformer(input_ids, attention_mask)

        text = self.text_linear(text_output.last_hidden_state)
        entities = self.entities_linear(self.transformer(entities_input_ids.view(fake_batch_size, -1), entities_attention_mask.view(fake_batch_size, -1)).pooler_output).view(batch_size, nb_entities, -1)
        attention = torch.einsum("blf,bef->bel", text, entities)

        # attention = dot product on each element of the batch, on the entities
        attention_entities_mask = entities_mask.unsqueeze(2) & attention_mask.unsqueeze(1) # make the softmax only on entities and words --> be->be1 & bl->b1l --> bel

        attention_max = torch.max(attention,dim=2,keepdim=True)[0] # max along sentence length
        attention_exp = torch.exp(attention-attention_max)
        attention_exp = attention_exp * attention_entities_mask.float()
        attention_softmax = attention_exp /( torch.sum(attention_exp,dim=2,keepdim=True) + (1-attention_entities_mask.float()) )

        # use this as query for image transformer output
        entity_attention_output = torch.sum(attention_softmax.unsqueeze(3) * text_output.last_hidden_state.unsqueeze(1), dim=2) # sum along sentence length: torch.Size([8, 10, 64, 1]) + torch.Size([8, 1, 64, 768] --> torch.Size([8, 10, 768])

        logits = self.classifier(entity_attention_output) # batch * entity * nb_labels (4)
        return self.loss(logits, labels)
