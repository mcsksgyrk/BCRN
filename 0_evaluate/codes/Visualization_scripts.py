import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, logm
import seaborn as sns

def kl_mvn(m0, S0, m1, S1):
    """
    https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv
    The following function computes the KL-Divergence between any two 
    multivariate normal distributions 
    (no need for the covariance matrices to be diagonal)
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    - accepts stacks of means, but only one S0 and S1
    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    # 'diagonal' is [1, 2, 3, 4]
    tf.diag(diagonal) ==> [[1, 0, 0, 0]
                          [0, 2, 0, 0]
                          [0, 0, 3, 0]
                          [0, 0, 0, 4]]
    # See wikipedia on KL divergence special case.              
    #KL = 0.5 * tf.reduce_sum(1 + t_log_var - K.square(t_mean) - K.exp(t_log_var), axis=1)   
                if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))                               
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N)

def kl_divergence_gaussians(mu0, cov0, mu1, cov1):
    """
    KL divergence D_KL(N0 || N1) between two multivariate Gaussians N0 ~ (mu0, cov0) and N1 ~ (mu1, cov1)
    """
    k = mu0.shape[0]
    cov1_inv = np.linalg.inv(cov1)
    diff = mu1 - mu0

    term1 = np.log(np.linalg.det(cov1) / np.linalg.det(cov0))
    term2 = np.trace(cov1_inv @ cov0)
    term3 = diff.T @ cov1_inv @ diff

    return 0.5 * (term1 - k + term2 + term3)

def plot_corr_matrix(df_corr, title="", figsize=(10, 10), vmin=-1, vmax=1, fontsize=7,
                     dropNa = False, save_fig=False, mask_it=False, show_diag=True):

    if dropNa:
        df_corr = df_corr.dropna(axis=0, how='all').dropna(axis=1, how='all')

    mask = None
    if mask_it:
        mask = np.triu(np.ones_like(df_corr, dtype=bool), k=0 if not show_diag else 1)

    fig = plt.figure(figsize=figsize)

    sns.heatmap(df_corr, 
                annot=False,
                mask=mask, 
                cmap='coolwarm', 
                center=0,
                vmin=vmin,
                vmax=vmax,
                xticklabels=True,
                yticklabels=True,
                square=True,
                linewidths=0.3,
                cbar_kws={"shrink": 0.8})
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    plt.title(title)
    plt.tight_layout()
    #fig1.patch.set_facecolor('#f8f6f6')
    if save_fig:
        plt.savefig(f"pics/{title}.pdf")
    plt.show()

def plot_corr_matrix_GPT(df_corr, title="", figsize=(10, 10), vmin=-1, vmax=1, fontsize=7,
                     dropNa=False, save_fig=False, mask_it=False, show_diag=True):

    if dropNa:
        df_corr = df_corr.dropna(axis=0, how='all').dropna(axis=1, how='all')

    mask = None
    if mask_it:
        mask = np.triu(np.ones_like(df_corr, dtype=bool), k=0 if not show_diag else 1)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.05, 0.05, 0.75, 0.9])  # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0, 0.1, 0.03, 0.8])  # [left, bottom, width, height] for colorbar

    sns.heatmap(df_corr, 
                ax=ax,
                annot=False,
                mask=mask, 
                cmap='coolwarm', 
                center=0,
                vmin=vmin,
                vmax=vmax,
                xticklabels=True,
                yticklabels=True,
                square=True,
                linewidths=0.3,
                cbar_ax=cbar_ax,
                cbar_kws={"shrink": 1})

    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsize)
    ax.set_title(title)
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"pics/{title}.pdf")

    plt.show()

def correlation_matrix_distance(A, B):
    """Correlation Matrix Distance (CMD), Herdin et al. (2005), https://doi.org/10.1049/el:20057319"""
    numerator = np.trace(A @ B)
    denominator = np.linalg.norm(A, 'fro') * np.linalg.norm(B, 'fro')
    return 1 - (numerator / denominator)

def rv_coefficient(A, B):
    """RV Coefficient, Escoufier (1973), https://doi.org/10.1016/0378-3758(78)90121-8"""
    numerator = np.trace(A @ B)
    denominator = np.sqrt(np.trace(A @ A) * np.trace(B @ B))
    return numerator / denominator

def safe_airm(A, B, epsilon=1e+1):
    """If COV() matrix is singular, https://doi.org/10.1016/S0047-259X(03)00096-4"""
    A_spd = A + epsilon * np.eye(A.shape[0])
    B_spd = B + epsilon * np.eye(B.shape[0])
    A_inv_sqrt = np.linalg.inv(sqrtm(A_spd))
    log_term = logm(A_inv_sqrt @ B_spd @ A_inv_sqrt)
    return np.linalg.norm(log_term, 'fro')

def affine_invariant_riemannian(A, B):
    """Affine-Invariant Riemannian Distance, https://doi.org/10.1007/s11263-005-3222-z"""
    try:
        A_inv_sqrt = np.linalg.inv(sqrtm(A))
        log_term = logm(A_inv_sqrt @ B @ A_inv_sqrt)
        return np.linalg.norm(log_term, 'fro')
    except:
        print("Matrix is singular, switching to linear shrinkage estimator formula")
        safe_airm(A, B)

def bures_wasserstein_distance(A, B):
    """Bures–Wasserstein Distance, https://doi.org/10.1016/0047-259X(82)90077-X"""
    sqrtA = sqrtm(A)
    middle = sqrtm(sqrtA @ B @ sqrtA)
    return np.sqrt(np.trace(A + B - 2 * middle))

def root_stein_divergence(A, B):
    """Root Stein Divergence, Sra, A New Metric on the Manifold of Kernel Matrices with Application to Matrix Geometric Means, NIPS, 2012"""
    sqrt_product = sqrtm(A @ B)
    return np.trace(A + B - 2 * sqrt_product)

def compute_matrix_distance(mat1, mat2, method='fro', mu1=None, mu2=None):
    if method == 'fro':
        return np.linalg.norm(mat1 - mat2, ord='fro')
    elif method == 'cmd':
        return correlation_matrix_distance(mat1, mat2)
    elif method == 'rv':
        return rv_coefficient(mat1, mat2)
    elif method == 'airm':
        return affine_invariant_riemannian(mat1, mat2)
    elif method == 'bures':
        return bures_wasserstein_distance(mat1, mat2)
    elif method == 'rsd':
        return root_stein_divergence(mat1, mat2)
    elif method == 'kl':
        return kl_mvn(mu1, mat1, mu2, mat2)
    else:
        raise ValueError("Unknown method. Use 'frobenius', 'cmd', 'rv', 'airm', 'skl', 'bures', or 'rsd'.")

def plot_it(distance_matrix, corr_xmls1, corr_xmls2, method, save_fig, title):
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(distance_matrix,
                xticklabels=corr_xmls1,
                yticklabels=corr_xmls2,
                annot=True,
                cmap='coolwarm',
                square=True,
                cbar_kws={'label': f'{method.upper()} Distance'})
    plt.title(title)
    plt.xlabel('Number of XMLs - Narrow')
    plt.ylabel('Number of XMLs - Wide')
    plt.tight_layout()
    #fig.patch.set_facecolor('#f8f6f6')
    if save_fig:
        plt.savefig(f"pics/{title}.pdf")
        plt.savefig(f"pics/{title}.png")
    plt.show()

def plot_correlation_matrix_triangle(corr_matrix: np.ndarray, labels=None, title="", 
                                     figsize=(12, 10), cmap='coolwarm', fmt=".2f", 
                                     show_diag=True, save_path=None):
    """
    Plot a lower triangular heatmap of a correlation matrix with annotations.
    """

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0 if not show_diag else 1)

    # Setup the plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                fmt=fmt,
                cmap=cmap,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.85, 'label': 'Correlation'},
                xticklabels=labels,
                yticklabels=labels,
                ax=ax)

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, rotation=0, fontsize=9)
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()