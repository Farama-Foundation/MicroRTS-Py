












import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
device = "cpu"

# no invalid action masking
print("=============regular=============")
target_logits = torch.tensor([1., 1., 1., 1.,] , requires_grad=True)
target_probs = Categorical(logits=target_logits)
action = target_probs.sample()
print(target_probs.probs)

# invalid action masking via logits
print("==================invalid action masking=============")
target_logits = torch.tensor([1., 1., 1., 1.,] , requires_grad=True)
# suppose action 1 is invalid
invalid_action_masks = torch.tensor([1., 1., 0., 1.,])
invalid_action_masks = invalid_action_masks.type(torch.BoolTensor)
adjusted_logits = torch.where(invalid_action_masks, 
                              target_logits, torch.tensor(-1e+8))
adjusted_probs = Categorical(logits=adjusted_logits)
action = adjusted_probs.sample()
print(adjusted_probs.probs)