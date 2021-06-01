import torch
import torch.nn.functional as F

class HSIC(object):
    def __init__(self, num_permutations: int, alpha: float = .05):
        """

        :param num_permutations: number of index permutations
        :param alpha: type 1 error level

        """

        self.num_permutations = num_permutations
        self.alpha = alpha

    @staticmethod
    def rbf(x: torch.Tensor, y: torch.Tensor, ls: float) -> torch.Tensor:
        """
        Calculates the RBF kernel in a vectorized form

        :param x: tensor of the first sample in the form of (num_samples, num_dim)
        :param y: tensor of the first sample in the form of (num_samples, num_dim)
        :param ls: lenght scale of the RBF kernel
        """

        # calc distances
        dists_sq = torch.cdist(x, y).pow(2)

        return torch.exp(-dists_sq / ls ** 2)

    def test_statistics(self, x: torch.Tensor, y: torch.Tensor, ls_x: float, ls_y: float) -> torch.Tensor:
        """
        Calculates the HSIC test statistics according to the code at
        http://www.gatsby.ucl.ac.uk/~gretton/indepTestFiles/indep.htm

        :param x: tensor of the first sample in the form of (num_samples, num_dim)
        :param y: tensor of the first sample in the form of (num_samples, num_dim)
        :param ls_x: lenght scale of the x RBF kernel
        :param ls_y: lenght scale of the y RBF kernel
        """

        num_samples = x.shape[0]

        # calculate the RBF kernel values
        kernel_x = self.rbf(x, x, ls_x)
        kernel_y = self.rbf(y, y, ls_y)

        H = torch.eye(num_samples) - torch.ones(num_samples, num_samples) / num_samples

        return torch.trace(H @ kernel_x @ H @ kernel_y) / num_samples ** 2

    @staticmethod
    def calc_ls(x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the lemght scale based on the median distance between points

        :param x: tensor of the first sample in the form of (num_samples, num_dim)
        """
        dists = F.pdist(x)

        return torch.median(dists)

    def run_test(self, x, y, device: str = 'cpu', ls_x: float = None, ls_y: float = None, bonferroni: int=1) -> bool:
        """
        Runs the HSIC test with randomly permuting the indices of y.

        :param x: tensor of the first sample in the form of (num_samples, num_dim)
        :param y: tensor of the second sample in the form of (num_samples, num_dim)
        :param ls_x: lenght scale of the x RBF kernel
        :param ls_y: lenght scale of the y RBF kernel
        :param bonferroni: Bonferroni correction coefficient (= #hypotheses)

        :return bool whether H0 (x and y are independent) holds
        """
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).unsqueeze(1).to(device).float()
        if not torch.is_tensor(y):
            y = torch.from_numpy(y).unsqueeze(1).to(device).float()

        if ls_x is None:
            ls_x = self.calc_ls(x)
        if ls_y is None:
            ls_y = self.calc_ls(y)

        alpha_corr = self.alpha / bonferroni

        num_samples = x.shape[0]

        # calculate test statistics for the permutations
        stats = []
        for _ in range(self.num_permutations):
            idx = torch.randperm(num_samples)

            stats.append(self.test_statistics(x, y[idx], ls_x, ls_y))

        stats = torch.tensor(stats)
        crit_val = torch.quantile(stats, 1 - self.alpha)

        p = (stats > crit_val).sum() / self.num_permutations

        print(f"p={p:.3f}, critical value={crit_val:.3f}")
        print(f"The null hypothesis (x and y is independent) is {p < alpha_corr}")

        return p < alpha_corr
