import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from datautils import *
from sparsegpt import *

DEV = 'cuda'

def get_dolly(model="databricks/dolly-v2-12b"):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto')
    return model

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

@torch.no_grad()
def dolly_sequential(model, dataloader, dev, args):
    print('Pruning ...')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.gpt_neox.layers

    model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(dev)
    model.gpt_neox.final_layer_norm = model.gpt_neox.final_layer_norm.to(dev)
    model.embed_out = model.embed_out.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.config.max_position_embeddings, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.gpt_neox.embed_in = model.gpt_neox.embed_in.cpu()
    model.gpt_neox.final_layer_norm = model.gpt_neox.final_layer_norm.cpu()
    model.embed_out = model.embed_out.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()
        for name in gpts:
            gpts[name].fasterprune(sparsity=args.sparsity, prunen=args.prenun, prunem=args.prenum)
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del gpts 
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache

@torch.no_grad()
def dolly_eval(model, testenc, dev, dataset: str):
    print('Evaluation...')

    testenc = testenc.input_ids
    nsamples = min(testenc.numel() // model.config.max_position_embeddings, args.nsamples)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.gpt_neox.layers

    model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(dev)
    model.gpt_neox.final_layer_norm = model.gpt_neox.final_layer_norm.to(dev)
    model.embed_out = model.embed_out.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.config.max_position_embeddings, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])

    for i in range(nsamples):
        batch = testenc[:, (i * model.config.max_position_embeddings):((i + 1) * model.config.max_position_embeddings)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.gpt_neox.embed_in = model.gpt_neox.embed_in.cpu()
    model.gpt_neox.final_layer_norm = model.gpt_neox.final_layer_norm.cpu()
    model.embed_out = model.embed_out.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu() 
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.gpt_neox.final_layer_norm = model.gpt_neox.final_layer_norm.to(dev)
    model.embed_out = model.embed_out.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        hidden_states = model.gpt_neox.final_layer_norm(hidden_states)
        lm_logits = model.embed_out(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.config.max_position_embeddings):((i + 1) * model.config.max_position_embeddings)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.config.max_position_embeddings
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.config.max_position_embeddings))
    print(f"Perplexity: {ppl.item():3f}")

    model.config.use_cache = use_cache

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    # databricks/dolly-v2-12b
    parser.add_argument(
        'model', type=str
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.', 
    )
    parser.add_argument(
        'prenun',
        type=int, default=1
    )
    parser.add_argument(
        'prenum',
        type=int, default=2
    )
    parser.add_argument(
        '--sparsity',
        type=float, default=0
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
       '--save', type=str, default='',
       help='Path to saved model.'
    )

    args = parser.parse_args()

    set_seed(args.seed)
    print('Model:', args.model)

    model = get_dolly(args.model)
    model.eval()

    dataloader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.config.max_position_embeddings
    )

    dolly_sequential(model, dataloader, DEV, args)
    total_params = sum(p.numel() for name, p in model.named_parameters() if 'dense' in name or 'query_key_value' in name)
    total_zero = sum((p == 0).sum().item() for name, p in model.named_parameters() if 'dense' in name or 'query_key_value' in name)
    print(f"Sparsity: {total_zero / total_params * 100:.2f}%")
    
    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.config.max_position_embeddings
        )
        print("Dataset:", dataset)
        dolly_eval(model, testloader, DEV, dataset)
        
    if args.save:
        model.save_pretrained(args.save)

    print('*'*17)
    print('='*17)
