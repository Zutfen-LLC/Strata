#!/usr/bin/env python3
"""Numerical acceptance for standalone HC; exact comparison remains diagnostic."""
import argparse, json, struct
from pathlib import Path
from compare_native_hc import compare
import native_hc_numeric as n


def evaluate(reference, candidate):
    exact = compare(reference, candidate)
    fixture = n.parse_fixture((reference / 'fixture.bin').read_bytes())
    n.need(json.loads((reference / 'report.json').read_text()).get('variant') == 3,
           'qualified selected native staged variant required')
    totals, weighted = n.source_inputs(fixture)
    xn, inject, violations, stages = {}, {}, [], []
    for backend, directory in [('native', reference), ('vulkan', candidate)]:
        def words(name):
            data = (directory / (name + '.f32le')).read_bytes()
            result = list(struct.unpack('<' + str(len(data) // 4) + 'I', data))
            for u in result:
                n.normal(u)
            return result
        rs, xn[backend], inject[backend] = words('rs'), words('normalized'), words('inject')
        bad_rs = [i for i, (r, s) in enumerate(zip(rs, totals)) if not n.rs_allowed(r, s, backend)]
        expected_xn, expected_inject = n.downstream(fixture, weighted, rs)
        bad_xn = [i for i, (a, b) in enumerate(zip(xn[backend], expected_xn)) if a != b]
        bad_dot = [i for i, (a, b) in enumerate(zip(inject[backend], expected_inject)) if a != b]
        for name, bad in [('rs_interval', bad_rs), ('weighted_association', bad_xn), ('ordered_dot', bad_dot if not bad_xn else [])]:
            if bad:
                violations.append({'backend': backend, 'check': name, 'count': len(bad), 'first': bad[0]})
        stages.append({'backend': backend, 'rs_interval': not bad_rs, 'normalized_order_exact': not bad_xn,
                       'injection_order_exact': None if bad_xn else not bad_dot})
    bounds = n.propagation(fixture, xn, inject) if not violations else []
    return {'contract': 'native-hc-normal-v1', 'qualified': False,
            'accepted_component_boundaries': not violations, 'exact_diagnostic': exact,
            'checks': stages, 'violations': violations, 'injection_absolute_bounds': bounds,
            'scope': 'fixed BF16 HC norm/injection, staged native variant, normal-or-zero domain; no full-model acceptance'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    try:
        result = evaluate(args.reference, args.candidate)
        with args.output.open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
        raise SystemExit(0 if result['accepted_component_boundaries'] else 1)
    except (OSError, ValueError, KeyError, struct.error) as error:
        print('STOP:', error, file=__import__('sys').stderr)
        raise SystemExit(2)
