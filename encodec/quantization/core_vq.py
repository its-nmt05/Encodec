import torch
import torch.nn as nn
import torch.nn.functional as F

import typing as tp

# exponential moving average implementation
def ema_inplace(moving_avg: torch.Tensor, new, decay: float):
    return moving_avg.mul_(decay).add_(new, alpha=(1 - decay))

def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)

def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t) # sample codebbok vectors from a kaiming uniform distribution 
    return t

# select random indices from the samples
def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device
    
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
        
    return samples[indices]

# implement kmeans algorithm for codebook initialization
def kmeans(samples, num_clusters: int, num_iters: int = 10):
    """sample (torch.Tensor): shape (N, dim)"""
    dim, dtype = samples.shape[-1], samples.dtype
    centroids = sample_vectors(samples, num_clusters)   # (K, dim)
    
    for _ in range(num_iters):
        diffs = samples.unsqueeze(-2) - centroids.unsqueeze(0)  # (N, 1, dim) - (1, K, dim) -> (N, K, dim) - (N, K, dim) after broadcasting
        dists = -(diffs ** 2).sum(-1)   # (N, K)
        
        buckets = dists.max(-1).indices # (N, K) -> (N,): containes the index of closest centroid for each sample
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)
        
        new_centroids = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_centroids.scatter_add_(0, (buckets.unsqueeze(-1).expand(-1, dim)), samples)
        new_centroids = new_centroids / bins_min_clamped[..., None]
        
        centroids = torch.where(zero_mask[..., None], centroids, new_centroids)
    
    return centroids, bins
        

class EuclideanCodebook(nn.Module):
    """Basic codebook with Euclidean distance"""
    
    def __init__(self, dim: int, codebook_size: int, kmeans_init: bool = True, kmeans_iters: int = 10,
                 decay: float = 0.99, epsilon: float = 1e-5, threshold_ema_dead_code: int = 2):
        super().__init__()
        # initialize codebook with kmeans or kaiming dist
        init_codebook: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not kmeans_init else torch.zeros   
        embed = init_codebook(codebook_size, dim) # codebook embedding
        
        self.decay = decay
        self.codebook_size = codebook_size
        self.k_means_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        
        # register all the buffers
        self.register_buffer("initiated", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    # initialize codebook vectors with kmeans
    def init_embed_(self, data):
        if self.initiated:
            return
        
        embed, cluster_size = kmeans(data, self.codebook_size, self.k_means_iters)  # cluster_size is the number of vectors in each cluster
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initiated.data.copy_(torch.Tensor([True]))
    
    def replace_(self, samples, mask):
        new_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        ) # replace codebook vectors with new samples
        self.embed.data.copy_(new_codebook)
    
    # replace codebook vectors which are not used
    def expire_codes_(self, batch_samples: torch.Tensor):
        if self.threshold_ema_dead_code == 0:   
            return
        
        expired_codes = self.cluster_size < self.threshold_ema_dead_code # check if codepoint is expired (not used)
        if not torch.any(expired_codes):
            return
        
        batch_samples = batch_samples.reshape(-1, batch_samples.shape[-1])
        self.replace_(batch_samples, mask=expired_codes)
        
    def preprocess(self, x):
        dim = x.shape[-1]
        x = x.reshape(-1, dim)
        return x
    
    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])
    
    def qunatize(self, x):
        embed = self.embed.t()  
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        ) # calculate the L2 distance between x and codebook vectors
        embed_ind = dist.max(-1).indices # closet codebook vector index
        return embed_ind
    
    def dequantize(self, embed_ind):
        return F.embedding(embed_ind, self.embed)
    
    def encode(self, x):
        shape = x.shape
        x = self.preprocess(x)
        emb_ind = self.qunatize(x)  # quantize x
        emb_ind = self.postprocess_emb(emb_ind, shape)
        return emb_ind
    
    def decode(self, embed_ind):
        return self.dequantize(embed_ind)
    
    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)
        
        self.init_embed_(x)
        embed_ind = self.qunatize(x)
        embed_onehot = F.one_hot(embed_ind, num_classes=self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantized = self.dequantize(embed_ind)
        
        if self.training:
            # update the codebook during training. Don't understand this part!!
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
        
        return quantized, embed_ind
    
    
class VectorQuantization(nn.Module):
    """Implement Vector Quantization"""

    def __init__(self, dim: int, codebook_size: int, codebook_dim: tp.Optional[int] = None, 
                 decay: float = 0.99, epsilon: float = 1e-5,
                 kmeans_init: bool = True, kmeans_iters: int = 50, 
                 threshold_ema_dead_code: int = 2, commitment_weight: float = 1.0):
        super().__init__()
        _codebook_dim: int = codebook_dim or dim
        requires_proj = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim) if requires_proj else nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim) if requires_proj else nn.Identity())
    
        self.eps = epsilon
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self._codebook = EuclideanCodebook(dim=_codebook_dim, codebook_size=codebook_size, 
                                           kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                                           decay=decay, epsilon=epsilon, 
                                           threshold_ema_dead_code=threshold_ema_dead_code)
        
    @property
    def codebook(self):
        return self._codebook.embed
        
    def encode(self, x):
        x = x.permute(0, 2, 1)  # (B, D, N) -> (B, N, D)
        x = self.project_in(x)
        return self._codebook.encode(x)
    
    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = quantize.permute(0, 2, 1) # (B, N, D) -> (B, D, N)
        return quantize
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, D, N) -> (B, N, D)
        x = self.project_in(x)
        
        quantized, embed_ind = self._codebook(x)    # quantized x, codebook index
        
        loss = torch.tensor([0.0], device=x.device, requires_grad=self.training)
        if self.training:
            quantized = x + (quantized - x).detach() # stop gradient flow during backprop through quantized
            
            if self.commitment_weight > 0: 
                commit_loss = F.mse_loss(x, quantized.detach())
                loss = loss + commit_loss * self.commitment_weight
                
        quantized = self.project_out(quantized)
        quantized = quantized.permute(0, 2, 1)  # (B, N, D) -> (B, D, N)
        return quantized, embed_ind, loss
    
    
class ResidualVectorQuantization(nn.Module):
    
    def __init__(self, *, num_quantizers: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantization(**kwargs) for _ in range(num_quantizers)])
        
    def forward(self, x,  n_q: tp.Optional[int] = None):
        residual = x
        quantized_out = 0.0
        
        all_losses = []
        all_indices = []
        
        n_q = n_q or len(self.layers)
        
        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out += quantized
            
            all_indices.append(indices)
            all_losses.append(loss)
        
        out_losses, out_indices = torch.stack(all_losses), torch.stack(all_indices)
        return quantized_out, out_indices, out_losses
    
    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices
    
    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)            
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out