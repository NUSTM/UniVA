import torch
import torch.nn as nn

def calculate_euclidean_distance(va_labels):
    va_labels_expanded = va_labels.unsqueeze(1).expand(-1, va_labels.size(0), -1) # [bsz, bsz, 2]
    va_labels_tiled = va_labels.repeat(va_labels.size(0), 1).view(va_labels.size(0), va_labels.size(0), -1) # [bsz, bsz, 2]
    difference = va_labels_expanded - va_labels_tiled
    euclidean_distance = torch.sqrt(torch.sum(difference ** 2, dim=2))
    return euclidean_distance

class VA_aware_SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='one', base_temperature=0.07):
        super(VA_aware_SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, multi_label=None, va_labels=None, thres_dict=None, batch_belong_diaID=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            multi_labels: multi-labels of shape [bsz, num_multi_labels]. 1 or 0
            va_labels: multidimensional emotional labels of shape [bsz, 2].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the similar multidimensional emotional labels (Valence and Arousal) as sample i.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                                'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if multi_label is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(features.device)
            if batch_belong_diaID is not None:
                for i in range(batch_size):
                    for j in range(batch_size):
                        if i != j and batch_belong_diaID[i] == batch_belong_diaID[j]:
                            mask[i, j] = 1

        if va_labels is not None:
            euclidean_distance = calculate_euclidean_distance(va_labels)
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j and mask[i, j] == 0 and euclidean_distance[i, j] < thres_dict :
                        # if batch_belong_diaID is None or batch_belong_diaID[i] != batch_belong_diaID[j]:
                        mask[i, j] = 1

        contrast_count = features.shape[1]  # n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #(bsz * n_views, feat_dim)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0] # only the first view
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature #(bsz * n_views, feat_dim)
            anchor_count = contrast_count 
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits 
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature) # (bsz, bsz)
        
        # for numerical stability 
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # (bsz, 1)
        logits = anchor_dot_contrast - logits_max.detach() # (bsz, bsz) set max_value in logits to zero

        # tile mask 
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            0
        ) # (anchor_cnt * bsz, contrast_cnt * bsz)

        mask = mask * logits_mask  # 1 indicates two items have similar multidimensional emotional labels and mask-out itself

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask # (anchor_cnt * bsz, contrast_cnt * bsz)

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)  

        # compute mean of log-likelihood over positive 
        if 0 in mask.sum(1):
            raise ValueError('Make sure there are at least two utterances with the common label')
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)

        # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
