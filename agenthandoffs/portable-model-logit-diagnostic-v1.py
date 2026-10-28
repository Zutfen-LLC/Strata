#!/usr/bin/env python3
"""Read-only retained-row diagnostics; no new acceptance criteria or inference."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

COMPARATOR = Path(__file__).resolve().parents[1] / 'tools/validation-harness/compare_logits.py'
spec = importlib.util.spec_from_file_location('frozen_compare_logits', COMPARATOR)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def analyze(reference, candidate):
    original = comparison.compare(reference, candidate)  # validates finite/matching rows
    n = len(reference)
    mean = lambda row: math.fsum(row) / n
    norm = lambda row: math.sqrt(math.fsum(v*v for v in row))
    mr, mc = mean(reference), mean(candidate)
    x = [v-mr for v in reference]
    y = [v-mc for v in candidate]
    nx, ny = norm(x), norm(y)
    centered = [b-a for a, b in zip(x, y)]
    covariance = math.fsum(a*b for a, b in zip(x, y))
    slope = covariance / (nx*nx) if nx else None
    residual = [b-slope*a for a, b in zip(x, y)] if slope is not None else None
    # Deterministic tie order agrees with the product's lowest-token-ID greedy rule.
    ranks = [sorted(range(n), key=lambda i: (-row[i], i)) for row in (reference, candidate)]
    probabilities = []
    for row in (reference, candidate):
        maximum = max(row)
        z = math.fsum(math.exp(v-maximum) for v in row)
        probabilities.append([math.exp(v-maximum)/z for v in row])
    top = []
    for row, order, probs in zip((reference, candidate), ranks, probabilities):
        top.append(dict(tokens=[dict(id=i, logit=row[i], probability=probs[i]) for i in order[:10]],
                        winning_logit_margin=row[order[0]]-row[order[1]] if n > 1 else None,
                        winning_probability_margin=probs[order[0]]-probs[order[1]] if n > 1 else None))
    return dict(schema='portable-model-logit-diagnostic-v1',
                scope='Descriptive diagnostics only; original gate is unchanged; no quality verdict.',
                original_limits=comparison.LIMITS, original_gate=original,
                diagnostics=dict(reference_mean=mr, candidate_mean=mc, mean_logit_delta=mc-mr,
                    reference_stddev=nx/math.sqrt(n), candidate_stddev=ny/math.sqrt(n),
                    centered_delta_rms=norm(centered)/math.sqrt(n),
                    centered_relative_l2=norm(centered)/max(1.0, nx),
                    correlation=covariance/(nx*ny) if nx and ny else None,
                    affine_candidate_from_reference=dict(slope=slope,
                        intercept=mc-slope*mr if slope is not None else None,
                        residual_rms=norm(residual)/math.sqrt(n) if residual is not None else None,
                        residual_relative_l2_reference_centered=norm(residual)/max(1.0, nx) if residual is not None else None),
                    reference_top=top[0], candidate_top=top[1],
                    candidate_rank_of_reference_winner=ranks[1].index(ranks[0][0])+1,
                    top10_overlap=len(set(ranks[0][:10]) & set(ranks[1][:10]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    # Existing bounded little-endian F32 reader: no raw arrays are printed or saved.
    result = analyze(comparison.read(args.reference), comparison.read(args.candidate))
    result['implementation_sha256'] = dict(
        diagnostic=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        comparator=hashlib.sha256(COMPARATOR.read_bytes()).hexdigest())
    print(json.dumps(result, indent=2, allow_nan=False))
    # Successful analysis is distinct from original_gate.numeric_pass in the JSON.


if __name__ == '__main__':
    main()
