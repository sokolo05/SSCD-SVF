import torch, torch.nn as nn

def param_groups_l2(model, backbone_wd=1e-5, head_wd=1e-3, no_decay_bn_bias=True):
    """
    Return a list of parameter groups ready to be passed to the optimizer.
    no_decay_bn_bias=True: no weight decay for BN weights and all biases.
    """
    backbone, head, no_decay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        *prefix, leaf_name = name.split('.')
        prefix = '.'.join(prefix)

        # 1. Classification head (fc / classifier)
        if 'fc' in prefix or 'classifier' in prefix:
            target = head
        else:
            target = backbone

        # 2. No-decay list: BN weights, all biases
        if no_decay_bn_bias and (leaf_name.endswith('bias') or 'bn' in prefix or 'BatchNorm' in prefix):
            no_decay.append(param)
        else:
            target.append(param)

    groups = [
        {'params': backbone, 'weight_decay': backbone_wd},
        {'params': head,     'weight_decay': head_wd},
        {'params': no_decay, 'weight_decay': 0.0}
    ]
    return groups

def param_groups_l2_dual(model, backbone_wd=1e-5, head_wd=1e-3, no_decay_bn_bias=True):
    """
    Perform layer-wise L2 regularization parameter grouping for DualModalModel.
    Returns groups ready to be passed to the optimizer.
    """
    backbone, head, no_decay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        *prefix, leaf = name.split('.'); prefix = '.'.join(prefix)

        # 1. Treat classification head / fusion layer as head
        if any(k in prefix for k in ('classifier', 'cross_attn')):
            target = head
        else:  # both backbones
            target = backbone

        # 2. No decay: bias + BN + classifier dropout weights (optional)
        if no_decay_bn_bias and (leaf.endswith('bias') or 'bn' in prefix or 'BatchNorm' in prefix):
            no_decay.append(param)
        else:
            target.append(param)

    groups = [
        {'params': backbone, 'weight_decay': backbone_wd},
        {'params': head,     'weight_decay': head_wd},
        {'params': no_decay, 'weight_decay': 0.0}
    ]
    return groups

# Label-smoothing loss
class LabelSmoothCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, x, target):
        log_probs = nn.functional.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
