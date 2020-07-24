












import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
device = "cpu"
action = 0
advantage = torch.tensor(1.)

# no invalid action masking
print("=============regular=============")
target_logits = torch.tensor([1., 1., 1., 1.,] , requires_grad=True)
target_probs = Categorical(logits=target_logits)
print("probs:   ", target_probs.probs)
log_prob = target_probs.log_prob(torch.tensor(action))
print("log_prob:", log_prob)
(log_prob*advantage).backward()
print("gradient:", target_logits.grad)

# invalid action masking via logits
print("==================invalid action masking=============")
target_logits = torch.tensor([1., 1., 1., 1.,] , requires_grad=True)
# suppose action 1 is invalid
invalid_action_masks = torch.tensor([1., 1., 0., 1.,])
invalid_action_masks = invalid_action_masks.type(torch.BoolTensor)
adjusted_logits = torch.where(invalid_action_masks, 
                              target_logits, torch.tensor(-1e+8))
adjusted_probs = Categorical(logits=adjusted_logits)
print("probs:   ", adjusted_probs.probs)
adjusted_log_prob = adjusted_probs.log_prob(torch.tensor(action))
print("log_prob:", adjusted_log_prob)
(adjusted_log_prob*advantage).backward()
print("gradient:", target_logits.grad)