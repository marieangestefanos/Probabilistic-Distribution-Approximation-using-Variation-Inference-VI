import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.stats import multivariate_normal
from scipy.special import gamma


#Generate new data
def new_data(mu, sigma, N):
    return np.random.normal(mu, sigma, N)


# True posterior distribution
def compute_p(mu_p, lbda_p, a_p, b_p, mu, tau):
    return (b_p ** a_p) * np.sqrt(lbda_p) / \
           (gamma(a_p) * np.sqrt(2 * np.pi)) \
           * tau ** (a_p - 0.5) * np.exp(-b_p*tau) \
           * np.exp(-0.5*lbda_p*(tau@((mu - mu_p) ** 2).T))


# Approximated posterior distribution
def compute_q(muN, lbdaN, aN, bN, mu, tau):
    q_mu = multivariate_normal(muN, 1.0/lbdaN).pdf(mu)
    q_tau = (1.0/gamma(aN)) * bN**aN *\
            tau**(aN-1) * np.exp(-bN * tau)
    return q_mu*q_tau


# Update parameters
def update_param(D, N, mu0, lbda0, b0, muN, lbdaN, aN, bN):
    #Compute the expectancies
    expec_mu = muN
    expec_mu2 = 1.0/lbdaN + muN**2
    expec_tau = aN / bN
    lbdaN = (lbda0 + N) * expec_tau
    bN = b0 - expec_mu*(np.sum(D) + lbda0*mu0) +\
         0.5*(np.sum(D ** 2) + lbda0*mu0**2 \
              + (lbda0+N)*expec_mu2)
    return lbdaN, bN


def viAlgo(N, mu, sigma):

    #Get D and x_mean values
    D = new_data(mu, sigma, N)
    x_mean = np.mean(D)

    #Initialization
    a0, b0, mu0, lbda0 = 0, 0, 0, 0
    mu = np.linspace(-2, 2, 100)
    tau = np.linspace(-2, 5, 100)

    #Parameters of the Gaussian-Gamma distrib
    mu_p = (lbda0*mu0 + N*x_mean)/(lbda0+N)
    lbda_p = lbda0+N
    a_p = a0 + N/2
    b_p = b0 + 0.5*np.sum((D-x_mean)**2)+\
          (lbda0*N*(x_mean-mu0)**2)/(2*(lbda0+N))

    #Update
    muN = (lbda0*mu0 + N*x_mean)/(lbda0+N)
    lbdaN = 0.1
    aN = a0 + (N+1)/2
    bN = 0.1

    lbdaOld = lbdaN
    bOld = bN

    p = compute_p(mu_p, lbda_p, a_p, b_p, mu[:, np.newaxis], \
                  tau[:, np.newaxis])

    for iter in range(NBITER_MAX):
        lbdaN, bN = update_param(D, N, mu0, lbda0, b0, muN, lbdaN, aN, bN)
        q = compute_q(muN, lbdaN, aN, bN, mu[:, np.newaxis], tau[:, np.newaxis])
        title = 'bN=%0.2f, lbdaN=%0.2f, N=%d, iter=%d'%(bN, lbdaN, N, iter)

        if ( (abs(bN - bOld) < thresh) and (abs(lbdaN - lbdaOld) < thresh) ):
            disp(mu, tau, p, q, title, 'green')
            plt.savefig("title")
            break
        else:
            disp(mu, tau, p, q, title, 'blue')
            plt.savefig("title")

        lbdaOld = lbdaN
        bOld = bN

    return muN, lbdaN, aN, bN


def disp(mu, tau, p, q, title, color):
    mm, tt = np.meshgrid(mu, tau)
    plt.figure()
    plt.contour(mm, tt, p)
    plt.contour(mm, tt, q, colors=color)
    # plt.axis("equal")
    plt.xlim([-2, 2])
    plt.ylim([-2, 5])
    plt.title(title)
    plt.show()

np.random.seed(122021)
NBITER_MAX = 300

N = 10
mu = 0
sigma = 1.0
thresh = 0.001

muN, lbdaN, aN, bN = viAlgo(N, mu, sigma)
print(f'muN={muN}\nlbdaN={lbdaN}\naN={aN}\nbN={bN}')