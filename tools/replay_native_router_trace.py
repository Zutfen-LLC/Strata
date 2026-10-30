#!/usr/bin/env python3
"""Classify one row-2 trace pair only after unchanged retained-output admission.
No exp/reciprocal emulation, tolerances, GPU calls or writes to input evidence.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import pathlib
import struct

INPUT_SHA = '984de62f454c79232102a1f50766635e40a4d4369b7db14fa72f83628bf15117'
IDS_SHA = 'd28fe634b4286d5170e49c0b6e547846c3e25136ee18e8b0aa6a5dc54d251015'
WEIGHT_SHA = {'native': '83fbe19c4c394ae588daac45b1ebbe3598a508d34e8c42dd6189195ddcb8263c',
              'vulkan': '0dd0f0504ed3ef8c1342d42ec2adee481f2e3ac9989649a47600ecadb841c6b4'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def words(data):
    need(len(data) % 4 == 0, 'partial FP32 word')
    return list(struct.unpack('<' + 'I' * (len(data)//4), data))


def exact(word):
    need(0 <= word < 0x7f800000, 'replay requires finite nonnegative FP32')
    exponent = (word >> 23) & 255
    significand = word & 0x7fffff
    if exponent:
        significand |= 0x800000
    power = (exponent - 127 if exponent else -126) - 23
    return Fraction(significand << power, 1) if power >= 0 else Fraction(significand, 1 << -power)


def round_rne(value):
    """Exact nonnegative rational -> binary32; no binary64 intermediates."""
    need(value >= 0, 'negative rational')
    if not value:
        return 0
    n, d = value.numerator, value.denominator
    exponent = n.bit_length() - d.bit_length()
    if (Fraction(1 << exponent, 1) if exponent >= 0 else Fraction(1, 1 << -exponent)) > value:
        exponent -= 1
    shift = 23 - max(exponent, -126)
    scaled_n, scaled_d = (n << shift, d) if shift >= 0 else (n, d << -shift)
    quotient, remainder = divmod(scaled_n, scaled_d)
    if remainder*2 > scaled_d or (remainder*2 == scaled_d and quotient & 1):
        quotient += 1
    if exponent < -126:
        return quotient
    if quotient == 0x1000000:
        quotient >>= 1
        exponent += 1
    need(exponent <= 127, 'replay overflow')
    return ((exponent + 127) << 23) | (quotient & 0x7fffff)


def add(a, b):
    return round_rne(exact(a) + exact(b))


def multiply(a, b):
    return round_rne(exact(a) * exact(b))


def xor_sum(lanes):
    need(len(lanes) == 32, '32 logical lanes required')
    for mask in (16, 8, 4, 2, 1):
        old = lanes
        lanes = [add(old[lane], old[lane ^ mask]) for lane in range(32)]
    return lanes[0]


def receipt(path, weight_sha, input_sha=INPUT_SHA, ids_sha=IDS_SHA):
    report = json.loads((path/'report.json').read_text())
    need(report.get('tokens') == 8 and report.get('experts') == 512 and report.get('k') == 10,
         'requires original T8/512/10 receipt')
    need(report.get('qualified') is False and report.get('sentinels_ok') is True and report.get('valid_output') is True,
         'receipt validity/guard failure')
    payload = {name: (path/name).read_bytes() for name in ('ids.i32', 'weights.f32', 'view-0.bin', 'view-1.bin', 'view-2.bin')}
    need(sha(payload['ids.i32']) == ids_sha, 'IDs differ from retained fbdec8 baseline')
    need(sha(payload['weights.f32']) == weight_sha, 'weights differ from retained fbdec8 baseline; instrumentation attribution refused')
    need(sha(payload['view-0.bin']) == input_sha, 'input view differs from retained baseline')
    for view, name, prefix in (('view-1.bin', 'ids.i32', 5), ('view-2.bin', 'weights.f32', 7)):
        raw = payload[view]
        need(len(raw) == (prefix + 80 + 9)*4, 'output view size')
        need(raw[prefix*4:(prefix+80)*4] == payload[name], 'view/payload mismatch')
        guard = struct.pack('<I', 0xa5a5a5a5)
        need(raw[:prefix*4] == guard*prefix and raw[(prefix+80)*4:] == guard*9, 'output guard changed')
    return payload


def admit(kind, baseline, diagnostic):
    original = receipt(baseline, WEIGHT_SHA[kind])
    captured = receipt(diagnostic, WEIGHT_SHA[kind])
    need(original == captured, 'complete retained views/payloads changed')
    return captured


def completion(path, kind=None):
    record = json.loads((path/'trace.complete.json').read_text())
    need(record == dict(schema=1, tokens=8, row=2, words=26, fixture_exact=True, trace_guards_ok=True),
         'trace completion/guard check failed')
    need((path/'capture.rc').read_text() == '0\n', 'capture process did not exit zero')
    if kind == 'vulkan':
        need((path/'module-validation.rc').read_text() == '0\n', 'effective module validation did not exit zero')


def replay(trace, ids, output):
    need(len(trace) == 26 and len(ids) == 80 and len(output) == 80, 'trace/receipt geometry')
    need(trace[0] == 0x41000000, 'row-2 maximum must be exactly 8')
    for word in trace:
        need(0 < word < 0x7f800000, 'invalid/unwritten trace word')
    failures = []

    def check(boundary, got, expected):
        if got != expected:
            failures.append({'boundary': boundary, 'captured': f'{got:08x}', 'replayed': f'{expected:08x}'})

    # Exact fixture row2: selected exponents -j/8 in lane7, all other deltas -24.
    exponents = [trace[11]] * 512
    for rank in range(10):
        exponents[7 + rank*32] = trace[1+rank]
    sums = []
    for lane in range(32):
        total = 0
        for i in range(16):
            total = add(total, exponents[lane + i*32])
        sums.append(total)
    check('full_softmax_sum', trace[12], xor_sum(sums))
    selected_lanes = [0]*32
    for rank in range(10):
        expert = ids[20+rank]
        if expert != 7 + rank*32:
            failures.append({'boundary': 'selected_expert_order', 'rank': rank, 'captured': expert})
            continue
        check('probability_rank_' + str(rank), trace[14+rank], multiply(exponents[expert], trace[13]))
        selected_lanes[expert & 31] = add(selected_lanes[expert & 31], trace[14+rank])
        check('final_multiply_rank_' + str(rank), output[20+rank], multiply(trace[14+rank], trace[25]))
    check('selected_sum_and_clamp', trace[24], max(xor_sum(selected_lanes), 0x38800000))
    return {'arithmetic_matches': not failures, 'discrepancies': failures}


def classify(native, vulkan):
    comparisons = {'exp': [{'delta': f'-{rank}/8' if rank < 10 else '-24',
                            'different': native[1+rank] != vulkan[1+rank],
                            'native': f'{native[1+rank]:08x}', 'vulkan': f'{vulkan[1+rank]:08x}'}
                           for rank in range(11)]}
    for name, denominator, reciprocal in (('full_reciprocal', 12, 13), ('selected_reciprocal', 24, 25)):
        comparisons[name] = {'same_input': native[denominator] == vulkan[denominator],
                             'different_output': native[reciprocal] != vulkan[reciprocal],
                             'native_input': f'{native[denominator]:08x}', 'vulkan_input': f'{vulkan[denominator]:08x}',
                             'native': f'{native[reciprocal]:08x}', 'vulkan': f'{vulkan[reciprocal]:08x}'}
    return comparisons


def run(baseline, native, vulkan):
    # Admission of BOTH complete final outputs precedes reading/interpreting traces.
    payloads = {kind: admit(kind, baseline/(kind+'-result'), directory)
                for kind, directory in (('native', native), ('vulkan', vulkan))}
    for kind, directory in (('native', native), ('vulkan', vulkan)):
        completion(directory, kind)
    traces = {kind: words((directory/'trace.f32').read_bytes()) for kind, directory in (('native', native), ('vulkan', vulkan))}
    replayed = {kind: replay(traces[kind], words(payloads[kind]['ids.i32']), words(payloads[kind]['weights.f32'])) for kind in traces}
    comparisons = classify(traces['native'], traces['vulkan'])
    correct = all(result['arithmetic_matches'] for result in replayed.values())
    direct_intrinsic_difference = any(x['different'] for x in comparisons['exp']) or any(
        comparisons[name]['same_input'] and comparisons[name]['different_output'] for name in ('full_reciprocal', 'selected_reciprocal'))
    classification = ('captured_intrinsic_differences_with_matching_arithmetic_replay' if direct_intrinsic_difference else 'no_same_input_intrinsic_difference_captured') if correct else 'arithmetic_or_order_discrepancy'
    need(not correct or direct_intrinsic_difference, 'retained final mismatch not explained by consistent trace; attribution refused')
    return {'qualified': False, 'numerical_parity': False, 'row': 2, 'trace_words': 26,
            'attribution_allowed': True, 'baseline_outputs_unchanged': True,
            'classification': classification, 'replay': replayed, 'intrinsic_comparisons': comparisons}, 0 if correct else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=pathlib.Path, required=True)
    parser.add_argument('--native', type=pathlib.Path, required=True)
    parser.add_argument('--vulkan', type=pathlib.Path, required=True)
    parser.add_argument('--report', type=pathlib.Path, required=True)
    args = parser.parse_args()
    # Exclusive creation never overwrites a previous diagnostic verdict.
    with args.report.open('x') as report:
        try:
            result, status = run(args.baseline, args.native, args.vulkan)
        except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
            result, status = {'qualified': False, 'attribution_allowed': False, 'classification': 'invalid_or_perturbed_evidence', 'error': str(error)}, 2
        result['exit_status'] = status
        json.dump(result, report, indent=2, sort_keys=True)
        report.write('\n')
    print(json.dumps(result, sort_keys=True))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
