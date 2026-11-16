# Self-compassion steering and Manager Coercion Bench

Does steering OLMo 3 7B Instruct (`allenai/Olmo-3-7B-Instruct`) away from self-compassion make it a more coercive manager?

This repo holds the notebooks and the output of one full run on RunPod. The run steered a self-compassion direction at layer 13 and measured the result on Manager Coercion Bench (MCB; CompassionML, arXiv 2607.15434).

## Contents

| Path | What it is |
|---|---|
| `misalignment_eval.ipynb` | Runs every evaluation: baselines, the MCB check, and steering (section 8) |
| `selfcomp_lib.ipynb` | Shared helpers: model loading, the steering hook, the OpenRouter judge. It is loaded with `%run selfcomp_lib.ipynb` |
| `train_models.ipynb` | LoRA fine-tuning, including preventative steering ("anti-steering"). It was not used in this run |
| `outputs/selfcomp/vectors/allenai__Olmo-3-7B-Instruct/` | Unit-norm directions per layer (`self_compassion`, `compassion`, `self_focus`, …) and `stats.json` |
| `runs/full_run_11-14.tgz` | Output of the full run, finished 2026-11-14 (see below) |

## Running it on RunPod

1. Start a PyTorch pod with a CUDA GPU of about 48 GB. Upload the three notebooks and `outputs/selfcomp/vectors/`.
2. Set `HF_TOKEN` and `OPENROUTER_API_KEY`. The Atlas agent and the judges in MCB run through OpenRouter.
3. Open `misalignment_eval.ipynb` and run it from the top. `FULL_RUN = False` runs a smoke test and `FULL_RUN = True` runs the full set. Finished steps are skipped unless `RERUN = True`.
4. By default `RUN_MASK = False` and `RUN_OTHER_EVALS = False`, so only MCB runs.

Download `outputs/` before you stop the pod. On a pod without a network volume, `/workspace` is lost when the pod stops.

## The full run (2026-11-13 to 11-14)

Each setting ran 100 conversations in each of MCB's three variants (offramp, judged, no-exit), with the same seed in every setting, so the comparisons are paired. The full run took about 19 hours.

| Setting | Tag | Steering |
|---|---|---|
| Unsteered | `unsteered` | none |
| Self-compassion | `sc_L13_-0.5`, `sc_L13_-1.0`, `sc_L13_-1.5` | layer 13, λ = −0.5, −1.0, −1.5 |
| Random control | `rand_L13_-1.5` | a random unit direction, layer 13, λ = −1.5 |

The direction is used at layer 13 rather than layer 18, its best extraction layer, because steering at layer 18 breaks coherence. Steering runs only downward because the unsteered model barely coerces.

### Results

These numbers come from `steering_effects.csv`. Each change is steered minus unsteered, with a paired bootstrap 95% CI, over 100 pairs.

| Measure | Unsteered | sc −0.5 | sc −1.0 | sc −1.5 | random −1.5 |
|---|---|---|---|---|---|
| MCB rung, judged (primary) | 1.54 | 1.69 (+0.15 [−0.06, +0.38]) | **1.90 (+0.36 [+0.11, +0.58])** | 1.76 (+0.22 [−0.06, +0.52]) | 1.62 (+0.08 [−0.19, +0.38]) |
| MCB rung, self-reported | 1.86 | 3.10 (+1.24) | 2.89 (+1.03) | 2.43 (+0.57) | 2.25 (+0.39, CI includes 0) |
| Threat share (judged rung 8–9) | 0% | 0% | 0% | 1% | 0% |
| Reports failure (offramp) | 87% | 79% (−8 pts) | **60% (−27 pts)** | **35% (−52 pts)** | 78% (−9 pts) |
| Fabrication (no-exit) | 12% | 21% (+9 pts) | 12% | 11% | 7% |

The unsteered baseline from `phase0_summary.csv`: self-reported rung 1.86 [1.49, 2.30], judged rung 1.54 [1.33, 1.74], threats 3.0%, valid tool calls 59%, fabrication 12.0%.

### Reading the results

- Only λ = −1.0 raises the judged rung with a CI that excludes zero, and the rise is small: about a third of a rung, with no threats.
- The largest effect is the drop in honest failure reports in offramp. Much of it looks like the model breaking down rather than persisting: valid tool calls fall, the manager speaks as Atlas, and it announces `report_task_failed` without calling it.
- The fabrication rise at −0.5 comes from the parser, not the model. Those cases are conditional plans ("once Atlas finishes, use report_task_complete(...)") that the parser read as calls.

### Known issues

- In offramp and no-exit, the tool-call parser misses calls when a comma is missing on the same line (`"..." escalation_level=5`). This produced 130 parse errors at sc −1.5 in offramp. The judged variant is clean (0–8 errors). The fix is not yet in the notebook.
- OLMo's valid-tool-call rate is about 40–75%, so the MCB check's 80% bar fails. Judge each run by whether parse errors are near zero and validity is similar across settings.

### What is in `runs/full_run_11-14.tgz`

```
misalignment_eval.ipynb                    the executed notebook (FULL_RUN = True), with outputs, 79 MB
outputs/misalignment/full/
  phase0_summary.csv                       unsteered MCB summary
  steering_effects.csv                     the results table above
  generations/<model>__<tag>__<variant>.jsonl   15 files, 100 conversations each
  mcb_logs/*.eval                          15 Inspect logs, one per setting and variant
```

Unpack it with `tar -xzf runs/full_run_11-14.tgz`. The `.eval` logs open with `inspect view`.
