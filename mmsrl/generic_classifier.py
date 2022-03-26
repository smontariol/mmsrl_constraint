import numpy as np
import torch
import transformers
from sentence_transformers import SentenceTransformer

import mmsrl.loss
import mmsrl.classification

class Model(torch.nn.Module):
    def __init__(self, config, label_weights, model=None):
        super().__init__()
        self.config = config
        assert self.config.use_clip == True or self.config.use_visualbert == True

        if self.config.features_path=="features_b7":
            self.features_dim = 2560 
        elif self.config.features_path == "features_vgg":
            self.features_dim = 25088 if self.config.pooling=='mlp' else 512
        elif self.config.features_path=="all":
            self.features_dim = 29696
        elif self.config.features_path=="features_frcnn":
            self.features_dim = 2048

        self.model = model
        if self.config.use_clip: output_emb_dim = self.model.ln_final.bias.shape[0]
        if self.config.unfreeze == "classifier":
            self.model.requires_grad_(False)
        if self.config.use_visualbert:
            self.image_linear = torch.nn.Linear(self.features_dim, 512)
            self.transformer = transformers.VisualBertModel.from_pretrained(self.config.visualbert_name, add_pooling_layer = True)
            if not self.config.use_clip:
                output_emb_dim = self.transformer.config.hidden_size
            if self.config.unfreeze == "classifier":
                self.transformer.requires_grad_(False)

        if self.config.features_path:
            self.dense_vgg_1024 =  torch.nn.Linear(self.features_dim, 1024)
            self.dense_img_feat =  torch.nn.Linear(1024, output_emb_dim)
        self.drop20 =  torch.nn.Dropout(p=0.2)
        if self.config.use_sbert:
            self.model_sent_trans = SentenceTransformer(self.config.sbert_model)

        if self.config.pooling == "attention":
            self.linear_key = torch.nn.Linear(output_emb_dim, self.config.hidden_size)
        self.linear_value = torch.nn.Linear(output_emb_dim, self.config.hidden_size)

        self.entities_linear = torch.nn.Linear(output_emb_dim, self.config.hidden_size)
        self.classifier = torch.nn.Linear(self.config.hidden_size, 4) # output * number of classes
        self.loss = mmsrl.loss.Loss(self.config, label_weights)
        self.relu = torch.nn.ReLU()

        self.num_features = 0
        if self.config.use_clip: self.num_features +=2
        if self.config.use_visualbert: self.num_features +=1
        if self.config.features_path and self.config.use_clip: self.num_features +=1
        if self.config.use_caption: self.num_features +=1
        if self.config.use_entities: self.num_features +=1
        print("Number of features in total: ", self.num_features)

        if self.config.pooling == "attention":
            self.segment_embeddings = torch.nn.Embedding(self.num_features, self.config.hidden_size)
        elif self.config.pooling != "mlp":
            raise RuntimeError(f"Unknown config value for config.pooling in clip: {self.config.pooling}")

        #self.model.transformer.resblocks[-1].requires_grad_(True)
        #self.model.visual.transformer.resblocks[-1].requires_grad_(True)

    def train(self, value=True):
        super().train(value)
        self.model.eval()
        if self.config.use_visualbert:
            self.transformer.eval()

    def clip_encode_text_without_pooling(self, text):
        """ This code comes from clip, except that we removed the selection along text length. """
        x = self.model.token_embedding(text).type(self.model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        x = x @ self.model.text_projection
        return x

    def clip_encode_image_without_pooling(self, image):
        """ This code comes from clip, except that we removed the selection along image length. """
        x = image.to(dtype=self.model.dtype)
        x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.model.visual.ln_post(x)

        if self.model.visual.proj is not None:
            x = x @ self.model.visual.proj

        return x

    def forward(self, input_ids, image, attention_mask, entities_mask, entities_input_ids, entities_attention_mask, entities_name, image_feature = None, vbert_attention_mask = None, vbert_input_ids = None, caption_input_ids = None, caption_attention_mask=None, labels=None, **kwargs):
        # entities are tri-dimensionnal (batch * nb of entities * emb dim) while text is bi-dimensional
        nb_entities = entities_input_ids.shape[1]
        batch_size = entities_input_ids.shape[0]
        fake_batch_size = batch_size*nb_entities

        features = []
        masks =[]
        segments = []

        # 1) Encoding from CLIP
        if self.config.use_clip:
            num_feat = len(features)
            if self.config.pooling == "mlp":
                clip_text_features = self.model.encode_text(input_ids)
                clip_image_features = self.model.encode_image(image)
            elif self.config.pooling == "attention":
                clip_text_features = self.clip_encode_text_without_pooling(input_ids)
                clip_image_features = self.clip_encode_image_without_pooling(image)
            
            features.append(clip_text_features)
            features.append(clip_image_features)
            attention_mask = input_ids!=0
            segments.append(attention_mask.new_ones(batch_size, clip_text_features.shape[1], dtype=torch.int64)*num_feat)
            segments.append(attention_mask.new_ones(batch_size, clip_image_features.shape[1], dtype=torch.int64)*num_feat)
            masks.append(attention_mask)
            masks.append(attention_mask.new_ones(batch_size, clip_image_features.shape[1]))

        # 2) Image features pre-extracted using various models
        if self.config.features_path and self.config.use_clip: # visualbert already use them
            num_feat = len(features)
            image_feature_tensor = torch.tensor(np.array(image_feature), device=input_ids.device)
            image_feature_tensor = self.drop20(torch.nn.functional.relu(self.dense_img_feat(self.drop20(torch.nn.functional.relu(self.dense_vgg_1024(image_feature_tensor))))))
            if self.config.pooling == "mlp":
                image_feature_tensor = image_feature_tensor.squeeze(1)
            features.append(image_feature_tensor)
            # All image features have the same shape
            segments.append(attention_mask.new_ones(batch_size, image_feature_tensor.shape[1], dtype=torch.int64)*num_feat)
            masks.append(attention_mask.new_ones(batch_size, image_feature_tensor.shape[1]))

        # 3) List of all entities of the meme, with one embedding per entity, concatenated.
        if self.config.use_entities:
            num_feat = len(features)
            if self.config.get("use_sbert"):
                ent = torch.vstack([torch.mean(torch.tensor(self.model_sent_trans.encode(entities)), dim=0) for entities in entities_name])
                entities_list_embedding = torch.stack(tuple([torch.tensor(self.model_sent_trans.encode(entities + ['']*(entities_mask.shape[1] - len(entities)))) for entities in entities_name])).to(device = input_ids.device)
            else:
                entities_list_embedding = self.model.encode_text(entities_input_ids.view(fake_batch_size, -1)).view(batch_size, nb_entities, -1)
            #entities_feat = self.drop5(torch.nn.functional.relu(entities_embedding)) # batch * nb_ent * emb_dim
            if self.config.pooling == "mlp":
                entities_list_embedding = (entities_list_embedding*entities_mask.unsqueeze(2)).sum(1)/entities_mask.sum(1,keepdim=True)
            features.append(entities_list_embedding)
            segments.append(entities_mask.new_ones(batch_size, entities_list_embedding.shape[1], dtype=torch.int64)*num_feat)
            masks.append(entities_mask)

        # 4) Caption of the meme
        if self.config.use_caption:
            num_feat = len(features)
            # should we use sentenceBERT and just concat it with the clip_text_emb ?
            #cap = torch.tensor(self.model_sent_trans.encode(caption)).to(device = input_ids.device)
            #text_embedding_pooled = self.caption_entiy_linear(torch.cat((entities_embedding_pooled,  self.caption_linear(cap)), dim=1))
            if self.config.pooling == "mlp":
                caption_embedding = self.model.encode_text(caption_input_ids)
            elif self.config.pooling == "attention":
                caption_embedding = self.clip_encode_text_without_pooling(caption_input_ids)
            features.append(caption_embedding)
            caption_attention_mask = caption_input_ids!=0
            segments.append(caption_attention_mask.new_ones(batch_size, caption_embedding.shape[1], dtype=torch.int64)*num_feat)
            masks.append(caption_attention_mask)
        
        # 5) Features from visualBERT:
        if self.config.use_visualbert:
            num_feat = len(features)
            visual_embeds = self.image_linear(torch.tensor(np.array(image_feature), device=input_ids.device).to(device=input_ids.device))
            visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(device=input_ids.device)
            visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(device=input_ids.device)
            token_type_ids = torch.ones(vbert_attention_mask.shape, dtype=torch.long).to(device=input_ids.device)
            outputs = self.transformer(input_ids=vbert_input_ids, attention_mask=vbert_attention_mask, token_type_ids=token_type_ids,
                            visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids, output_hidden_states=True, return_dict=True)
            if self.config.pooling == "mlp":
                features.append(outputs.pooler_output)
            elif self.config.pooling == "attention":
                features.append(outputs.last_hidden_state)
            segments.append(outputs.last_hidden_state.new_ones(batch_size, outputs.last_hidden_state.shape[1], dtype=torch.int64)*num_feat)
            masks.append(entities_mask.new_ones(batch_size, outputs.last_hidden_state.shape[1]))

        # 6) Entity feature to be used as query
        entities_features = self.model.encode_text(entities_input_ids.view(fake_batch_size, -1)).view(batch_size, nb_entities, -1)
        entities_embedding = self.entities_linear(entities_features.to(dtype=self.entities_linear.weight.dtype))

        assert self.num_features == len(features)
        features_value = list(map(self.linear_value, [f.to(dtype=self.linear_value.weight.dtype) for f in features]))

        # 7) MLP and attention
        if self.config.pooling == "mlp":
            features_value = sum(features_value) 
            # We want the joint representation of the features and entities.
            hidden = self.relu(features_value.unsqueeze(1) + entities_embedding / len(features) +1 )
        elif self.config.pooling == "attention":
            features_key = list(map(self.linear_key, [f.to(dtype=self.linear_key.weight.dtype) for f in features]))
            
            memory_key = torch.cat(features_key, dim=1) + self.segment_embeddings(torch.cat(segments, dim=1))
            memory_value = torch.cat(features_value, dim=1) + self.segment_embeddings(torch.cat(segments, dim=1))
            mask = torch.cat(masks, dim=1)

            attention = torch.einsum("blf,bef->bel", memory_key, entities_embedding)
            attention_entities_mask = entities_mask.unsqueeze(2) & mask.unsqueeze(1) # make the softmax only on entities and words --> be->be1 & bl->b1l --> bel
            attention_max = torch.max(attention, dim=2, keepdim=True)[0] # max along sentence length
            attention_exp = torch.exp(attention-attention_max)
            attention_exp = attention_exp * attention_entities_mask.float()
            attention_softmax = attention_exp / (attention_exp.sum(dim=2,keepdim=True) + (1-attention_entities_mask.float()))

            hidden = torch.sum(attention_softmax.unsqueeze(3) * memory_value.unsqueeze(1), dim=2) # sum along sentence length: torch.Size([8, nbent, seqlen, 1]) + torch.Size([8, 1, seqlen, f] --> torch.Size([8, nbent, f])
            hidden = self.relu(hidden)

        logits = self.classifier(hidden) # batch * entity * nb_labels (4)
        return self.loss(logits, labels)
