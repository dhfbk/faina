import logging # @A
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__) # @A

from machamp.model.machamp_decoder import MachampDecoder


class MachampClassificationDecoder(MachampDecoder, torch.nn.Module):
    def __init__(self, task, vocabulary, input_dim, device, loss_weight: float = 1.0, 
                 decoder_dropout: float = 0.0, topn: int = 1,
                 metric: str = 'accuracy', **kwargs):
        super().__init__(task, vocabulary, loss_weight, metric, device, **kwargs)

        self.nlabels = len(self.vocabulary.get_vocab(task))
        self.hidden_to_label = torch.nn.Linear(input_dim, self.nlabels)
        self.hidden_to_label.to(device)
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)
        self.topn = topn
        self.device = device
        self.class_weights = kwargs["class_weights"] if "class_weights" in kwargs.keys() else None # @A
        self.class_weights_counts = kwargs["class_weights_counts"] if "class_weights_counts" in kwargs.keys() else None # @A

        self.decoder_dropout = torch.nn.Dropout(decoder_dropout)
        self.decoder_dropout.to(device)

    def forward(self, mlm_out, mask, gold=None):
        if self.decoder_dropout.p > 0.0:
            mlm_out =  self.decoder_dropout(mlm_out) 

        # @A ---
        # Compute class weights for cross entropy loss if the class weights parameter 
        # is set for the task (executed at the first forward call only for efficiency)
        if self.class_weights is not None:
            # If class weights are explicitly defined in the config file, use them
            if type(self.class_weights) is dict:
                if len(self.class_weights.values()) > (self.nlabels):
                    logger.error(f"ERROR. Class weights must be {self.nlabels}, but {len(self.class_weights.values())} are defined.")
                
                weights = [0.0] * (self.nlabels)
                for label, weight in self.class_weights.items():
                    weights[label] = weight
                self.class_weights = torch.FloatTensor(weights).to(self.device)
                self.loss_function = torch.nn.CrossEntropyLoss(
                    reduction='sum', ignore_index=-100, weight=self.class_weights)

            # If class weights are set to balanced, compute and set the weights automatically
            elif (self.class_weights == "balanced"):
                num_samples = sum(self.class_weights_counts.values())

                weights = [0.0] * (self.nlabels)
                for label, label_count in self.class_weights_counts.items():
                    weights[label] = num_samples / float((self.nlabels-1) * label_count)
                self.class_weights = torch.FloatTensor(weights).to(self.device)
                self.loss_function = torch.nn.CrossEntropyLoss(
                    reduction='sum', ignore_index=-100, weight=self.class_weights)

            # Class weights are already initialized
            else:
                pass
        # @A ---

        logits = self.hidden_to_label(mlm_out)
        out_dict = {'logits': logits}
        if type(gold) != type(None):
            maxes = torch.add(torch.argmax(logits[:, 1:], 1), 1)
            self.metric.score(maxes, gold, self.vocabulary.inverse_namespaces[self.task])
            if self.additional_metrics:
                for additional_metric in self.additional_metrics:
                    additional_metric.score(maxes, gold, None, self.vocabulary.inverse_namespaces[self.task])
            out_dict['loss'] = self.loss_weight * self.loss_function(logits, gold)
        return out_dict

    def get_output_labels(self, mlm_out, mask, gold=None):
        logits = self.forward(mlm_out, mask, gold)['logits']
        if self.topn == 1:
            maxes = torch.add(torch.argmax(logits[:, 1:], 1), 1)
            return {'sent_labels': [self.vocabulary.id2token(label_id, self.task) for label_id in maxes]}
        else:
            labels = []
            probs = []
            class_probs = F.softmax(logits, -1)
            for sent_scores in class_probs:
                topn = min(self.nlabels, self.topn)
                topk = torch.topk(sent_scores[1:], topn)
                labels.append([self.vocabulary.id2token(label_id + 1, self.task) for label_id in topk.indices])
                probs.append([score.item() for score in topk.values])
            return {'sent_labels': labels, 'probs': probs}
