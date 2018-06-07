import torch
from traj2vec.utils.math import normal_log_pdf

def mse_loss(output, target):
    return (output - target).pow(2).mean()

def vae_loss(output, target, embed_mu, embed_logvar, out_mean, out_logvar):
    var = out_logvar.exp_()
    SE = (output - target).pow(2)

    RCE_element = 0.5 * (torch.sum(torch.log(var), 1) + torch.sum(SE / (var), 1))
    RCE = torch.mean(RCE_element)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = embed_mu.pow(2).add_(embed_logvar.exp()).mul_(-1).add_(1).add_(embed_logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)

    return RCE + KLD

def reinforce_loss(output, traj_info):
    # actions is (batch_size, seq_len, action_dim)
    actions, means, logvars, returns = traj_info['actions'], traj_info['means'], traj_info['logvars'], traj_info['disc_returns']
    logpdf = normal_log_pdf(means, logvars, actions)
    return torch.mean(logpdf * torch.autograd.Variable(returns))