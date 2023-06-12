# SparseLLM
Applying Sparsity to LLMs and exploring whether they can be structurally pruned as a result.
Mainly using SparseGPT: https://arxiv.org/abs/2301.00774
Also, worth looking into LoRA: https://arxiv.org/pdf/2106.09685.pdf

## Sparse LLMs
### Models
Dolly 3B, Dolly 7B, Dolly 12B

### Datasets
c4, wikitext, ptb

### Sparsity Patterns
Unstructured sparsity: (--sparsity > 0)
N:M semi-structured sparsity: (--prunen > 0 and --prunem > 0) --> For every M elements, N of them are guaranteed to be zeros.

### How to prune the model

python prune_dolly.py databricks/dolly-v2-12b c4 1 2 --save sparse_dolly_12/

This will apply 1:2 semi-structured sparsity to Dolly 12B using c4 dataset for pruning and save the sparse model to sparse_dolly_12/
Currently, it runs on CPU, but it takes time. You can use smaller models if you want faster experimenting time or if you don't have enough memory


### Required Libraries
torch: tested on v1.10.1+cu111
transformers: tested on v4.21.2
datasets: tested on v1.17.0
