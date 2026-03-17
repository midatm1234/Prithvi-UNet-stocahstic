# CORDEX v4 Evaluation Summary

Date: 2026-03-17
Repo: `/mnt/data2/kyo/granite-wxc`

## Completed checks

- Python syntax checks: **PASS**
- CORDEX YAML parse checks: **PASS**
- v4 YAML backup presence (`*_v4.yaml`): **PASS**
- `runs_v4` backup policy (no prediction artifacts): **PASS**
- `pr=0` log1p normalize/inverse round-trip and non-negativity guard: **PASS**

Validation artifact:
- `examples/CORDEX_ML/evaluations/v4_validation_summary.json`

## Runtime evaluation status

Model-level baseline-vs-v4 metric evaluation script is available:
- `examples/CORDEX_ML/utils/evaluate_v4_outputs.py`

Current execution in this environment is blocked by missing runtime dependencies:
- `torch` unavailable
- `numpy` unavailable

Because of this blocker, the following metrics were **not executed here**:
- seam visibility ratio (baseline vs v4)
- precipitation zero/near-zero fraction comparison
- heavy-tail (`p95/p99`) comparison
- RMSE deltas (including `tasmax` degradation check)

## Next run command (when runtime deps are available)

```bash
cd /mnt/data2/kyo/granite-wxc
python examples/CORDEX_ML/utils/evaluate_v4_outputs.py \
  --baseline-config examples/CORDEX_ML/NZ_T1_ACCESS-CM2_static.yaml \
  --v4-config examples/CORDEX_ML/NZ_T1_ACCESS-CM2_static_v4.yaml \
  --checkpoint examples/CORDEX_ML/runs_v4/NZ_T1_ACCESS-CM2_static_train/NZ_T1_ACCESS-CM2_static/checkpoints/best.ckpt \
  --num-samples 64 \
  --batch-size 1 \
  --output-json examples/CORDEX_ML/evaluations/NZ_T1_static_v4_eval.json
```
