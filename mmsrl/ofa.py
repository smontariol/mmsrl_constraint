import math

import torch

import mmsrl.loss
import mmsrl.utils


class Model(torch.nn.Module):
    def __init__(self, config, label_weights, ofa_task, ofa_config, ofa_models):
        super().__init__()
        self.config = config
        self.ofa_task = ofa_task
        self.ofa_config = ofa_config
        assert(len(ofa_models) == 1) # Check if it's really an ensemble or a single model
        self.model = ofa_models[0]

        if self.config.get("ofa_classification_head"):
            self.model.register_classification_head("mmsrl", 4)
            self.loss = mmsrl.loss.Loss(self.config, label_weights)
        else:
            self.loss = torch.nn.NLLLoss(reduction="none")

        if self.config.unfreeze != "all":
            self.model.requires_grad_(False)

        if self.config.unfreeze == "classifier":
            if self.config.get("ofa_classification_head"):
                self.model.classification_heads["mmsrl"].requires_grad_(True)
            else:
                self.model.decoder.output_projection.requires_grad_(True)

        if self.config.get("freeze_embeddings"):
            self.model.encoder.embed_tokens.requires_grad_(False)
            self.model.decoder.embed_tokens.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.config.get("freeze_resnet"):
            for module in self.model.encoder.embed_images.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
                    module.requires_grad_(False)

    def masked_log_softmax(self, logits, mask):
        if mask is None:
            return torch.nn.functional.log_softmax(logits, dim=2)

        fake_batch_size: int = logits.shape[0]
        sequence_length: int = logits.shape[1]
        vocabulary_size: int = logits.shape[2]
        mask = mask.view(fake_batch_size, sequence_length, vocabulary_size)

        logits_max = torch.max(logits, dim=2, keepdim=True)[0] # max along vocabulary
        logits_exp = torch.exp(logits - logits_max)
        logits_exp = mask.float() * logits_exp
        logits_sum = torch.sum(logits_exp, dim=2, keepdim=True)
        logits_sum += (logits_sum==0).float() # avoid log(0) when everything is masked
        log_softmax = logits - logits_max - torch.log(logits_sum)
        return mask.float() * log_softmax

    def forward(self, input_ids, length, image, image_mask, decoder_input_ids, decoder_target, decoder_target_attention_mask, labels=None, **kwargs):
        if self.config.get("ofa_classification_head"):
            if self.config.get("group_image_entities"):
                batch_size: int = input_ids.shape[0]
                num_entities: int = input_ids.shape[1]
                fake_batch_size: int = batch_size * num_entities
                input_ids = input_ids.view(fake_batch_size, -1)
                length = length.view(fake_batch_size)
                decoder_input_ids = decoder_input_ids.view(fake_batch_size, -1)
                image = image.repeat_interleave(repeats=num_entities, dim=0)
                image_mask = image_mask.repeat_interleave(repeats=num_entities, dim=0)
            else:
                input_ids = input_ids.squeeze(1)
                length = length.squeeze(1)
                decoder_input_ids = decoder_input_ids.squeeze(1)

            outputs = self.model(
                    input_ids,
                    length,
                    decoder_input_ids,
                    patch_images=image,
                    patch_masks=image_mask[:, 0, 0],
                    classification_head_name="mmsrl")[0]
            if input_ids.shape[0] == 1:
                # fix for stupid wild squeeze from OFA/fairseq.
                outputs = outputs.unsqueeze(0)
            if self.config.get("group_image_entities"):
                logits = outputs.view(batch_size, num_entities, 4)
            else:
                # When using OFA without group_image_entities, we always have 1 entity per sample
                logits = outputs.unsqueeze(1)
            return self.loss(logits, labels)

        batch_size: int = input_ids.shape[0]
        inner_input_size: int = input_ids.shape[1]
        fake_input_batch_size: int = batch_size * inner_input_size

        encoded = self.model.encoder(
                input_ids.view(fake_input_batch_size, -1),
                src_lengths=length.view(fake_input_batch_size),
                patch_images=image.repeat_interleave(inner_input_size, 0),
                # FIXME build patch_masks the same way patches are built from image.
                patch_masks=image_mask.repeat_interleave(inner_input_size, 0)[:, 0, 0])

        inner_output_size: int = decoder_input_ids.shape[1]
        fake_output_batch_size: int = batch_size * inner_output_size
        decode_per_encode: int = inner_output_size // inner_input_size
        decoder_sequence_length: int = decoder_input_ids.shape[2]

        encoded["encoder_out"] = [encoded["encoder_out"][0].repeat_interleave(decode_per_encode, 1)]
        encoded["encoder_padding_mask"] = [encoded["encoder_padding_mask"][0].repeat_interleave(decode_per_encode, 0)]
        encoded["position_embeddings"] = [encoded["position_embeddings"][0].repeat_interleave(decode_per_encode, 0)]

        decoded = self.model.decoder(
                decoder_input_ids.view(fake_output_batch_size, decoder_sequence_length),
                encoder_out=encoded)
        lm_logits = self.masked_log_softmax(decoded[0], kwargs.get("decoder_constraint"))

        fake_loss_batch_size: int = fake_output_batch_size * decoder_sequence_length
        lm_loss = self.loss(lm_logits.view(fake_loss_batch_size, -1), decoder_target.view(fake_loss_batch_size))
        lm_loss = lm_loss.view(fake_output_batch_size, decoder_sequence_length)
        loss = lm_loss

        if self.config.get("label_smoothing"):
            loss *= (1 - self.config.label_smoothing)
            mask = kwargs["decoder_constraint"].view(fake_output_batch_size, decoder_sequence_length, -1)
            num_classes = mask.float().sum(2, keepdim=True)
            num_classes += (num_classes == 0)
            uniform = mask.float() / num_classes
            loss += self.config.label_smoothing * - torch.sum(uniform * lm_logits, 2)

        if self.config.get("encouraging_loss"):
            bonus = torch.log(-torch.expm1(lm_logits))
            log_end = self.config.encouraging_loss
            if log_end != 1:
                probs = torch.exp(lm_logits)
                y_log_end = math.log1p(-log_end)
                bonus_after_log_end = 1 / (log_end - 1) * (probs - log_end) + y_log_end
                bonus = torch.where(probs > log_end, bonus_after_log_end, bonus)
            ec_loss = self.loss(bonus.view(fake_loss_batch_size, -1), decoder_target.view(fake_loss_batch_size)).view(fake_output_batch_size, decoder_sequence_length)

            if self.config.get("label_smoothing"):
                ec_loss *= (1 - self.config.label_smoothing)
                mask = kwargs["decoder_constraint"].view(fake_output_batch_size, decoder_sequence_length, -1)
                num_classes = mask.float().sum(2, keepdim=True)
                num_classes += (num_classes == 0)
                uniform = mask.float() / num_classes
                ec_loss += self.config.label_smoothing * - torch.sum(uniform * bonus, 2)

            loss += ec_loss

        if self.config.get("only_train_answer"):
            loss *= kwargs["decoder_answer_mask"].view(fake_output_batch_size, decoder_sequence_length)
        loss = loss.sum(1)
        if self.config.get("loss_mean_sentence_length"):
            loss /= decoder_target_attention_mask.view(fake_output_batch_size, decoder_sequence_length).sum(1)
        loss = loss.mean()

        if self.config.get("only_score_answer"):
            lm_loss *= kwargs["decoder_answer_mask"].view(fake_output_batch_size, decoder_sequence_length)
        lm_loss = lm_loss.sum(1)

        if self.config.ofa_task == "vqa_gen" and decode_per_encode == 4:
            logits = -lm_loss.view(batch_size, decode_per_encode)
            prediction = torch.nn.functional.softmax(logits, dim=1).unsqueeze(1)
        elif self.config.ofa_task == "snli_ve" and decode_per_encode in [2, 3]:
            assert(self.config.get("neither_for_other")) # For another commit
            assert(self.config.maybe_is not in ["yes", "no"]) # For another commit too =(
            # logits is of size: batch × label × answer:
            #     - label comes from the encoding of "entity is a hero", "entity is a villain", etc.
            #     - answer comes from the SNLI question answer: "no", "maybe" or "yes"
            logits = -lm_loss.view(batch_size, inner_input_size, decode_per_encode)

            # FIXME just compare the probabilities of "yes" for now
            prediction = torch.nn.functional.softmax(logits[:, :, -1], dim=1).unsqueeze(1)
        else:
            prediction = None

        return loss, prediction
