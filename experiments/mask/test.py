# suppose action 1 is invalid

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
action = 0
advantage = torch.tensor(1.)
device = "cpu"

# no invalid action masking
print("=============regular=============")
target_logits = torch.tensor([1., 1., 1., 1.,] , requires_grad=True)
target_probs = Categorical(logits=target_logits)
log_prob = target_probs.log_prob(torch.tensor(action))
print("log_prob", log_prob)
(log_prob*advantage).backward()
print("gradient", target_logits.grad)
print()

# invalid action masking via logits
print("==================invalid action masking=============")
target_logits = torch.tensor([1., 1., 1., 1.,] , requires_grad=True)
invalid_action_masks = torch.tensor([1., 1., 0., 1.,])
invalid_action_masks = invalid_action_masks.type(torch.BoolTensor)
adjusted_logits = torch.where(invalid_action_masks, target_logits, torch.tensor(-1e+8))
adjusted_probs = Categorical(logits=adjusted_logits)
adjusted_log_prob = adjusted_probs.log_prob(torch.tensor(action))
print("log_prob", adjusted_log_prob)
(adjusted_log_prob*advantage).backward()
print("gradient", target_logits.grad)
print()

# invalid action masking via importance sampling
print("==================regular importance sampling=============")
target_logits = torch.tensor([1., 1., 1., 1.,] , requires_grad=True)
target_probs = Categorical(logits=target_logits)
invalid_action_masks = torch.tensor([1., 1., 0., 1.,])
invalid_action_masks = invalid_action_masks.type(torch.BoolTensor)
adjusted_logits = torch.where(invalid_action_masks, target_logits, torch.tensor(-1e+8))
adjusted_probs = Categorical(logits=adjusted_logits)
log_prob = target_probs.log_prob(torch.tensor(action))
adjusted_log_prob = adjusted_probs.log_prob(torch.tensor(action))

importance_sampling = target_probs.probs[torch.tensor(action)] / (adjusted_probs.probs[torch.tensor(action)])
print("log_prob", log_prob)
(importance_sampling.detach()*log_prob*advantage).backward()
print("gradient", target_logits.grad)
print()


# invalid action masking via logits
print("==================invalid action masking=============")
target_logits = torch.tensor([1., 1., 1., 1.,] , requires_grad=True)
invalid_action_masks = torch.tensor([1., 1., 0., 1.,])
invalid_action_masks = invalid_action_masks.type(torch.BoolTensor)
adjusted_logits = torch.where(invalid_action_masks, target_logits, torch.tensor(-2.))
adjusted_probs = Categorical(logits=adjusted_logits)
adjusted_log_prob = adjusted_probs.log_prob(torch.tensor(action))
print("adjusted_probs", adjusted_probs.probs)
(adjusted_log_prob*advantage).backward()
print("gradient", target_logits.grad)
print()

# no invalid action masking with different parameterization
print("=============regular but differrent parameterization=============")
target_logits = torch.tensor([1., 1., -2., 1.,] , requires_grad=True)
target_probs = Categorical(logits=target_logits)
log_prob = target_probs.log_prob(torch.tensor(action))
print("target_probs", target_probs.probs)
(log_prob*advantage).backward()
print("gradient", target_logits.grad)
print()
new_target_logits = target_logits + target_logits.grad
new_target_probs = Categorical(logits=new_target_logits)
print("target_probs", new_target_probs.probs)


