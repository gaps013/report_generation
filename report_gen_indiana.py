from combined_decoder import CombinedDecoder
from torch import nn
import functools
from typing import Any, Dict
import copy
import numpy as np
import torch
from torch.nn import functional as F
from image_model import Image_Model
from beam_search import AutoRegressiveBeamSearch

class ReportGeneration(nn.Module):
    def __init__(self, device, image_model, visual_feature_size, use_beam_search=True,
                 max_sequence_length=256, sos_index=1, eos_index=2, embedding_dim=512,
                 vocab_size=1000, num_layers=6, attention_heads=3, drop_out=0.1, padding_idx=0,
                 number_of_classes=41, beam_size=5):
        super(ReportGeneration, self).__init__()
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.attention_heads = attention_heads
        self.drop_out = drop_out
        self.padding_idx = padding_idx
        self.number_of_classes = number_of_classes
        self.image_model = Image_Model(image_model, visual_feature_size)
        self.use_beam_search = use_beam_search
        self.combined_model = CombinedDecoder(device=self.device, vocab_size=self.vocab_size, embedding_dim=self.embedding_dim,
                                              num_layers=self.num_layers, attention_heads=self.attention_heads,
                                              drop_out=self.drop_out, max_sequence_length=self.max_sequence_length,
                                              padding_idx=self.padding_idx)
        self.image_feature_projection = nn.Linear(self.image_model.visual_feature_size, self.embedding_dim)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        self.beam_search = AutoRegressiveBeamSearch(self.eos_index, beam_size=beam_size,
                                                    max_steps=self.max_sequence_length)
    def forward(self, inputs):
        images = inputs['images']
        image_features = self.image_model(images)
        batch_size = image_features.size(0)
        image_features = image_features.view(inputs["images"].size(0), self.image_model.visual_feature_size, -1).permute(0, 2, 1)
        projected_image_features = self.image_feature_projection(image_features)
        if(not inputs['inference']):
            # labels = inputs['labels']
            tokenised_report = inputs['tokenised_report']
            actual_report = inputs['actual_report']
            report_length = inputs['report_length']
            output_logits = self.combined_model(projected_image_features,
                                                actual_report, report_length)
            loss = self.loss(output_logits[:, :-1].contiguous().view(-1, self.combined_model.vocab_size),
                tokenised_report[:, 1:].contiguous().view(-1),)
            output_dict: Dict[str, Any] = {
                'loss': loss,
                # Single scalar per batch for logging in training script.
                'loss_components': {'report_loss': loss.clone().detach()},
            }
            if not self.training:
                # During validation (while pretraining), get best prediction
                # at every time-step.
                output_dict['predictions'] = torch.argmax(output_logits, dim=-1)

        else:
            if(not self.use_beam_search):
                start_predictions = projected_image_features.new_full(
                    (batch_size,self.max_sequence_length), self.sos_index
                ).long().to(self.device)

                report_length = torch.Tensor(np.full((start_predictions.size(0),), self.max_sequence_length)).to(self.device)

                print('Report Length: ', report_length.shape)
                output_logits = self.combined_model(projected_image_features,
                                                    start_predictions, report_length)
                output_dict = {"predictions": torch.argmax(output_logits, dim=-1)}
                return output_dict

            start_predictions = projected_image_features.new_full(
                (batch_size,), self.sos_index
            ).long()
            beam_search_step = functools.partial(
                self.beam_search_step, projected_image_features
            )
            all_top_k_predictions, _ = self.beam_search.search(
                start_predictions, beam_search_step
            )
            best_beam = all_top_k_predictions[:, 0, :]
            output_dict = {"predictions": best_beam}

        return output_dict

    def beam_search_step(self, projected_image_features: torch.Tensor, partial_reports: torch.Tensor) -> torch.Tensor:
            batch_size, num_features, combined_feature_size = (
                projected_image_features.size()
            )
            beam_size = int(partial_reports.size(0) / batch_size)
            if beam_size > 1:
                projected_image_features = projected_image_features.unsqueeze(1).repeat(
                    1, beam_size, 1, 1
                )
                projected_image_features = projected_image_features.view(
                    batch_size * beam_size, num_features, combined_feature_size
                )

            # Provide caption lengths as current length (irrespective of predicted
            # EOS/padding tokens). shape: (batch_size, )
            report_lengths = torch.ones_like(partial_reports)
            if len(report_lengths.size()) == 2:
                report_lengths = report_lengths.sum(1)
            else:
                # Add a time-step. shape: (batch_size, 1)
                partial_reports = partial_reports.unsqueeze(1)
            # shape: (batch_size * beam_size, partial_caption_length, vocab_size)
            output_logits = self.combined_model(projected_image_features, partial_reports, report_lengths)
            # Keep features for last time-step only, we only care about those.
            output_logits = output_logits[:, -1, :]
            # Return logprobs as required by `AutoRegressiveBeamSearch`.
            # shape: (batch_size * beam_size, vocab_size)
            next_logprobs = F.log_softmax(output_logits, dim=1)
            # Set logprobs of last predicted tokens as high negative value to avoid
            # repetition in caption.
            for index in range(batch_size * beam_size):
                next_logprobs[index, partial_reports[index, -1]] = -10000000
            return next_logprobs