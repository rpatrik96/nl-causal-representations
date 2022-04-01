from care_nl_ica.metrics.dep_mat import JacobianBinnedPrecisionRecall
import torch


def test_jacobian_prec_recall():
    num_dim = 3
    num_thresholds = 15
    jac_pr = JacobianBinnedPrecisionRecall(num_thresholds=num_thresholds)
    target = (
        (
            torch.tril(torch.bernoulli(0.5 * torch.ones(num_dim, num_dim)), -1)
            + torch.eye(num_dim)
        )
        .bool()
        .float()
    )
    preds = torch.tril(torch.randn_like(target))
    jac_pr(preds, target)
    precisions, recalls, thresholds = jac_pr.compute()
    print(precisions, recalls, thresholds)
