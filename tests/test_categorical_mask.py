import torch
import numpy as np
from torch.distributions.categorical import Categorical

device = "cpu"

class CategoricalMasked(Categorical):

    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = torch.BoolTensor(masks).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)

single_logits = torch.tensor([[1., 1., 1., 1.,]] , requires_grad=True)
single_invalid_action_masks = np.array([[0, 0, 1, 1]])

scm = CategoricalMasked(logits=single_logits, masks=single_invalid_action_masks)
print(scm.probs)

multiple_logits = torch.tensor([[1., 1., 1., 1.,], [1., 2., 1., 2.,]] , requires_grad=True)
multiple_invalid_action_masks = np.array([[0, 0, 1, 1], [0, 1, 1, 1]])

mcm = CategoricalMasked(logits=multiple_logits, masks=multiple_invalid_action_masks)
print(mcm.probs)