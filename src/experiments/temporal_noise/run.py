"""Paired canonical GPT experiments on stationary Markov trace corruption."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from handcoded.gates import make_gate_names, phi
from src.data.boolean_circuit_tasks import _coherent_wrong_mask
from src.data.dataclass import ANSWER_SEP
from src.data.datasets import ContinuationDataset
from src.data.registry import TASKS
from src.data.sample import generate_unique
from src.experiments.reliability.review_corpus_validation import table_metrics
from src.training.optim import build_gpt, make_optimizer
from src.training.seed import configure_device, set_seed

CONDITIONS = [(0.3, 0., False), (0.3, .5, False), (0.3, 1., False),
              (.7, 0., False), (.7, 1., False), (1., 0., False),
              (.3, 0., True), (.3, 1., True)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def gates_of(inst):
    return [step.split('>')[0] for step in inst.correct_trace.split()]


def make_corpus(instances, rho, persistence, seed):
    """C_t copies C_(t-1) with probability kappa, otherwise redraws Bern(rho)."""
    n, depth = len(instances), len(gates_of(instances[0]))
    rng = np.random.default_rng(seed)
    draws, copies = rng.random((n, depth)), rng.random((n, depth))
    clean = np.zeros((n, depth), dtype=bool)
    clean[:, 0] = draws[:, 0] < rho
    for t in range(1, depth):
        clean[:, t] = np.where(copies[:, t] < persistence, clean[:, t-1], draws[:, t] < rho)
    names = make_gate_names()
    ids = {g: i for i, g in enumerate(names)}
    counts = np.zeros((len(names), 16, 16), dtype=np.int64)
    traces, states = [], np.empty((n, depth), dtype=np.int64)
    for i, inst in enumerate(instances):
        state, steps = int(inst.prompt[1:5], 2), []
        for t, gate in enumerate(gates_of(inst)):
            truth = phi(state, gate)
            nxt = truth if clean[i, t] else truth ^ _coherent_wrong_mask(gate)
            counts[ids[gate], state, nxt] += 1
            assert (nxt == truth) == clean[i, t]
            steps.append(f'{gate}>{nxt:04b}')
            states[i, t] = nxt
            state = nxt
        trace = ' '.join(steps)
        assert (trace == inst.correct_trace) == bool(clean[i].all())
        traces.append(trace)
    return traces, states, clean, counts


def make_dataset(instances, task, traces, answer_loss):
    # All conditions process the same number of tokens. In the no-answer arm,
    # both the terminal values and their entire supervised suffix are removed.
    targets = [f' {trace}{ANSWER_SEP}{inst.gold if answer_loss else "0000"}\n'
               for inst, trace in zip(instances, traces)]
    data = ContinuationDataset(instances, task.tokenizer, task.block_size, targets)
    if not answer_loss:
        for i, (inst, trace) in enumerate(zip(instances, traces)):
            suffix = len(inst.prompt) + 1 + len(trace)
            data.mask[i, suffix-1:] = False
    return data


def batch_schedule(n, steps, seed, batch=128):
    generator = torch.Generator().manual_seed(seed)
    chunks = []
    while len(chunks) < steps:
        chunks.extend(torch.randperm(n, generator=generator)[:n//batch*batch].reshape(-1, batch))
    return torch.stack(chunks[:steps])


@torch.inference_mode()
def score_strings(model, tokenizer, strings, spans, device, batch=128):
    """Joint full-vocabulary probabilities of designated character spans."""
    results = []
    for begin in range(0, len(strings), batch):
        texts, regions = strings[begin:begin+batch], spans[begin:begin+batch]
        width = max(map(len, texts)) - 1
        x = torch.full((len(texts), width), tokenizer.pad_id, device=device, dtype=torch.long)
        y = x.clone()
        for i, text in enumerate(texts):
            encoded = torch.tensor(tokenizer.encode(text), device=device)
            x[i, :len(encoded)-1], y[i, :len(encoded)-1] = encoded[:-1], encoded[1:]
        logits, _ = model(x)
        logp = logits.float().log_softmax(-1).gather(-1, y[..., None]).squeeze(-1)
        for i, region in enumerate(regions):
            results.append([float(logp[i, start-1:end-1].sum()) for start, end in region])
    return np.asarray(results)


@torch.inference_mode()
def conditional_metrics(model, task, instances, device):
    texts, spans = [], []
    for inst in instances:
        text = f'{inst.prompt} {inst.correct_trace}'
        offset = len(inst.prompt)+1
        regions = []
        for step in inst.correct_trace.split():
            start = offset+step.index('>')+1
            regions.append((start, start+4)); offset += len(step)+1
        texts.append(text); spans.append(regions)
    logs = score_strings(model, task.tokenizer, texts, spans, device)
    # Full candidate-state scoring at step two: same prompt, either correct or
    # coherent-wrong first state, then evaluate ALL 16 successors of that state.
    candidate_texts, candidate_spans = [], []
    targets = np.empty((len(instances), 2), dtype=int)
    for i, inst in enumerate(instances):
        gates = gates_of(inst)
        first_true = phi(int(inst.prompt[1:5], 2), gates[0])
        for branch in range(2):
            first = first_true if branch == 0 else first_true ^ _coherent_wrong_mask(gates[0])
            prefix = f'{inst.prompt} {gates[0]}>{first:04b} {gates[1]}>'
            targets[i, branch] = phi(first, gates[1])
            for state in range(16):
                candidate_texts.append(prefix+f'{state:04b}')
                candidate_spans.append([(len(prefix), len(prefix)+4)])
    candidate_logs = score_strings(model, task.tokenizer, candidate_texts, candidate_spans, device, 256).reshape(-1, 2, 16)
    mass = np.exp(candidate_logs)
    selected = np.take_along_axis(mass, targets[..., None], axis=-1)[..., 0]
    norm = mass.sum(-1)
    metrics = {f'gold_probability_step_{t+1}': float(np.exp(logs[:, t]).mean()) for t in range(logs.shape[1])}
    metrics.update({f'gold_nll_step_{t+1}': float(-logs[:, t].mean()) for t in range(logs.shape[1])})
    for branch, label in enumerate(['correct_prefix', 'wrong_prefix']):
        metrics[f'{label}_step2_probability'] = float(selected[:, branch].mean())
        metrics[f'{label}_step2_conditional_probability'] = float((selected[:, branch]/norm[:, branch]).mean())
        metrics[f'{label}_step2_accuracy'] = float((candidate_logs[:, branch].argmax(-1) == targets[:, branch]).mean())
        metrics[f'{label}_step2_valid_mass'] = float(norm[:, branch].mean())
    metrics['prefix_probability_gap'] = metrics['correct_prefix_step2_probability']-metrics['wrong_prefix_step2_probability']
    return metrics, logs, candidate_logs, targets


@torch.inference_mode()
def rollout(model, task, instances, device, anchor=False, batch=128):
    tokenizer = task.tokenizer
    records = []
    for begin in range(0, len(instances), batch):
        subset = instances[begin:begin+batch]
        contexts = [inst.prompt+(' '+inst.correct_trace.split()[0] if anchor else '') for inst in subset]
        lengths = torch.tensor([len(s) for s in contexts], device=device)
        initial = lengths.clone()
        buffer = torch.full((len(subset), task.block_size), tokenizer.pad_id, dtype=torch.long, device=device)
        for i, context in enumerate(contexts):
            buffer[i, :len(context)] = torch.tensor(tokenizer.encode(context), device=device)
        rows = torch.arange(len(subset), device=device)
        for _ in range(task.max_new_tokens):
            logits, _ = model(buffer[:, :int(lengths.max())])
            nxt = logits[rows, lengths-1].argmax(-1)
            buffer[rows, lengths] = nxt
            lengths += 1
        for i, inst in enumerate(subset):
            tail = tokenizer.decode(buffer[i, int(initial[i]):int(lengths[i])].tolist()).split('\n', 1)[0]
            text = (inst.correct_trace.split()[0] if anchor else '')+tail
            found = re.findall(r'([xcst][0-3]{1,3})>([01]{4})(?![01])', text)
            gold_steps = [tuple(s.split('>')) for s in inst.correct_trace.split()]
            depth = len(gold_steps)
            # Require the full syntactic trace, not just any state strings found.
            trace_part = text.split(':', 1)[0].strip()
            tokens = trace_part.split()[:depth]
            parsed = []
            for token in tokens:
                match = re.fullmatch(r'([xcst][0-3]{1,3})>([01]{4})', token)
                parsed.append(match.groups() if match else None)
            valid = len(parsed) == depth and all(p is not None and p[0] == g[0] for p,g in zip(parsed, gold_steps))
            exact = valid and parsed == gold_steps
            endpoint = valid and parsed[-1][1] == inst.gold
            final_match = re.search(r':\s*([01]{4})(?![01])', text)
            records.append(dict(prompt=inst.prompt, anchored=anchor, generated=text,
                valid_trace=bool(valid), exact_trace=bool(exact), endpoint_correct=bool(endpoint),
                terminal_answer_correct=bool(final_match and final_match[1] == inst.gold),
                state_steps_correct=sum(p == g for p,g in zip(parsed, gold_steps)),
                regex_steps=len(found)))
    metrics = {name: float(np.mean([r[key] for r in records])) for name,key in
               [('valid_trace','valid_trace'),('exact_trace','exact_trace'),('endpoint_accuracy','endpoint_correct'),
                ('terminal_answer_accuracy','terminal_answer_correct')]}
    return metrics, records


def run_condition(args, seed, condition):
    rho, persistence, answer_loss = condition
    tag = f's{seed}_rho{rho:g}_k{persistence:g}_answer{int(answer_loss)}'
    path = args.output/tag
    if (path/'metrics.json').exists():
        print('Already completed:', tag, flush=True); return
    path.mkdir(parents=True, exist_ok=False)
    task = TASKS[f'boolean_circuit_{args.depth}']
    train_seed = 501+seed-2001
    train = generate_unique(task, args.train_size, train_seed)
    evaluation = generate_unique(task, args.eval_size, 101, {i.prompt for i in train})
    traces, states, clean, counts = make_corpus(train, rho, persistence, 777+seed-2001)
    data = make_dataset(train, task, traces, answer_loss)
    schedule = batch_schedule(len(train), args.steps, 12345)
    names = make_gate_names(); ids = {g:i for i,g in enumerate(names)}
    truth = np.array([[phi(s,g) for s in range(16)] for g in names])
    starts = np.array([int(i.prompt[1:5],2) for i in evaluation])
    operations = np.array([[ids[g] for g in gates_of(i)] for i in evaluation])
    answers = np.array([int(i.gold,2) for i in evaluation])
    tabular, q, margins = table_metrics(counts, truth, starts, operations, answers)
    n00, n01 = (~clean[:,:-1]).sum(), clean[:,:-1].sum()
    protocol = dict(tag=tag, seed=seed, rho=rho, persistence=persistence, answer_loss=answer_loss,
        train_seed=train_seed, depth=args.depth, train_size=len(train), eval_size=len(evaluation),
        steps=args.steps, architecture='canonical character GPT: 2 blocks, 4 heads, width 128, dropout 0',
        batch_size=128, lr=.0003, weight_decay=0, grad_clip=1., precision='bf16 training, float32 evaluation, TF32 enabled',
        empirical_local_correctness=float(clean.mean()), empirical_whole_trace_correctness=float(clean.all(1).mean()),
        correctness_per_position=clean.mean(0).tolist(),
        empirical_clean_after_clean=float((clean[:,:-1]&clean[:,1:]).sum()/n01) if n01 else None,
        empirical_clean_after_wrong=float((~clean[:,:-1]&clean[:,1:]).sum()/n00) if n00 else None,
        theoretical_clean_after_clean=persistence+(1-persistence)*rho,
        theoretical_clean_after_wrong=(1-persistence)*rho,
        train_prompt_sha256=sha('\n'.join(i.prompt for i in train).encode()),
        eval_prompt_sha256=sha('\n'.join(i.prompt for i in evaluation).encode()),
        corpus_sha256=sha(data.x.numpy().tobytes()+data.y.numpy().tobytes()+data.mask.numpy().tobytes()),
        schedule_sha256=sha(schedule.numpy().tobytes()), script_sha256=sha(Path(__file__).read_bytes()), tabular=tabular,
        device=args.device, model_init_sha256=None)
    np.savez_compressed(path/'corpus.npz', counts=counts, q=q, margins=margins, clean=clean, states=states, truth=truth)
    (path/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    device = configure_device(args.device, compile=False, bf16=True, distributed=False)
    set_seed(seed)
    config = SimpleNamespace(embedding=128, heads=4, layers=2, dropout=0., lr=.0003, weight_decay=0.)
    model = build_gpt(task, config, device)
    protocol['model_init_sha256'] = sha(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters()))
    (path/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    optimizer = make_optimizer(model, config, device)
    x,y,mask = data.x.to(device).long(), data.y.to(device).long(), data.mask.to(device).float()
    schedule = schedule.to(device)
    start = time.monotonic(); losses=[]
    model.train()
    for step, indices in enumerate(schedule,1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _,loss = model(x[indices], y[indices], mask[indices])
        if not torch.isfinite(loss):
            raise FloatingPointError((tag,step))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        if step == 1 or step % 1000 == 0 or step == args.steps:
            item=dict(step=step,loss=float(loss),elapsed_seconds=time.monotonic()-start)
            losses.append(item); print(tag,json.dumps(item),flush=True)
        if step == args.steps//2:
            torch.save(dict(model=model.state_dict(),step=step),path/'midpoint.pt')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),step=args.steps),path/'final.pt')
    (path/'training.json').write_text(json.dumps(losses,indent=2)+'\n')
    model.eval()
    metrics, logs, candidate_logs, candidate_targets = conditional_metrics(model,task,evaluation,device)
    np.savez_compressed(path/'conditional_predictions.npz',gold_state_logp=logs,candidate_logp=candidate_logs,targets=candidate_targets)
    for anchor in [False,True]:
        extra,records = rollout(model,task,evaluation,device,anchor)
        label = 'anchored' if anchor else 'ordinary'
        metrics.update({f'{label}_{k}':v for k,v in extra.items()})
        (path/f'{label}_generations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    metrics.update(tag=tag,seed=seed,rho=rho,persistence=persistence,answer_loss=answer_loss,
                   total_seconds=time.monotonic()-start, **{f'tabular_{k}':v for k,v in tabular.items()})
    if not answer_loss:
        metrics['ordinary_terminal_answer_accuracy']=None
        metrics['anchored_terminal_answer_accuracy']=None
    (path/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('COMPLETE',json.dumps(metrics),flush=True)
    del model,optimizer,x,y,mask
    torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seeds',type=int,nargs='+',default=[2001,2002,2003])
    parser.add_argument('--steps',type=int,default=8000)
    parser.add_argument('--depth',type=int,default=8)
    parser.add_argument('--train-size',type=int,default=20000)
    parser.add_argument('--eval-size',type=int,default=1000)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--conditions',type=int,nargs='+',default=list(range(len(CONDITIONS))))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    jobs=[(seed,CONDITIONS[i]) for seed in args.seeds for i in args.conditions]
    for index,(seed,condition) in enumerate(jobs):
        if index % args.shards == args.shard:
            run_condition(args,seed,condition)


if __name__=='__main__':
    main()
