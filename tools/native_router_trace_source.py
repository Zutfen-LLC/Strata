#!/usr/bin/env python3
"""Generate isolated row-2 trace variants from hash-pinned baseline sources.
Only stores and named FP32 results of existing expressions are added.
"""
import argparse
import hashlib
import pathlib
import re

PINS = {'cuda': '89131c17bb9809e7ed02d08d2e3b03d2a2b0e077885d2f82df9471a7b8bf2256',
        'vulkan': '373c88f4ce7c530ea84f42607c18a620ae965142a52c1c7482dd63103ef2b08d'}


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('router trace source anchor changed: ' + old)
    return source.replace(old, new)


def stores(code):
    return '\n/* ROUTER_TRACE_BEGIN */\n' + code + '\n/* ROUTER_TRACE_END */\n'


def cuda_body(source):
    begin = source.index('__launch_bounds__(256, 1)\n__global__ void route_multi(')
    end = source.index('// The same contract for other (experts, top-k)', begin)
    return source[begin:end]


def generate(kind, source):
    if hashlib.sha256(source.encode()).hexdigest() != PINS[kind]:
        raise ValueError('router trace requires the pinned fbdec8 baseline source')
    if kind == 'cuda':
        body = cuda_body(source)
        body = once(body, 'void route_multi(', 'void route_multi_diagnostic(')
        body = once(body, 'float* __restrict__ weights, int n_tok)', 'float* __restrict__ weights, int n_tok, float* trace)')
        body = once(body, '    maximum = warp_max(maximum);', '    maximum = warp_max(maximum);' + stores('    if (tk == 2 && lane == 0) trace[0] = maximum;'))
        body = once(body, '        values[i] = expf(values[i] - maximum);', '        values[i] = expf(values[i] - maximum);' + stores('    if (tk == 2 && lane == 7 && i < 10) trace[1+i] = values[i];\n    if (tk == 2 && lane == 0 && i == 0) trace[11] = values[i];'))
        body = once(body, '    const float reciprocal = 1.0f / warp_sum(sum);', '    const float trace_full_sum = warp_sum(sum);\n    const float reciprocal = 1.0f / trace_full_sum;' + stores('    if (tk == 2 && lane == 0) { trace[12] = trace_full_sum; trace[13] = reciprocal; }'))
        body = once(body, '        if ((expert & 31) == lane) {', stores('    if (tk == 2 && lane == 0) trace[14+rank] = best;') + '        if ((expert & 31) == lane) {')
        body = once(body, '    const float inverse_selected_sum = 1.0f / selected_sum;', '    const float inverse_selected_sum = 1.0f / selected_sum;' + stores('    if (tk == 2 && lane == 0) { trace[24] = selected_sum; trace[25] = inverse_selected_sum; }'))
        helper = source[source.index('__device__ __forceinline__ float warp_sum('):source.index('__launch_bounds__(256, 1)')]
        license_text = source[:source.index('#include')]
        return license_text + '#pragma once\n#include <cuda_runtime.h>\n#include <cfloat>\n#include <cstddef>\n#include <cstdint>\nnamespace strata_router_diagnostic {\n' + helper + body + '}\n'
    body = once(source, 'buffer Status {uint invalid;}', 'buffer Status {uint invalid;float trace[26];}')
    body = once(body, '    maximum=xor_max(maximum,lane);', '    maximum=xor_max(maximum,lane);' + stores('    if(token==2 && lane==0)trace[0]=maximum;'))
    body = once(body, '        values[i]=exp(shifted);', '        values[i]=exp(shifted);' + stores('    if(token==2 && lane==7 && i<10)trace[1+i]=values[i];\n    if(token==2 && lane==0 && i==0)trace[11]=values[i];'))
    body = once(body, '    precise float reciprocal=1.0/xor_sum(sum,lane);', '    precise float trace_full_sum=xor_sum(sum,lane);\n    precise float reciprocal=1.0/trace_full_sum;' + stores('    if(token==2 && lane==0){trace[12]=trace_full_sum;trace[13]=reciprocal;}'))
    body = once(body, '        if((expert&31)==int(lane)) {', stores('    if(token==2 && lane==0)trace[14+rank]=best;') + '        if((expert&31)==int(lane)) {')
    body = once(body, '    precise float inverse_selected_sum=1.0/selected_sum;', '    precise float inverse_selected_sum=1.0/selected_sum;' + stores('    if(token==2 && lane==0){trace[24]=selected_sum;trace[25]=inverse_selected_sum;}'))
    return body


def restore(kind, generated):
    """Remove diagnostic additions; tests require byte-identical baseline body."""
    body = re.sub(r'\n/\* ROUTER_TRACE_BEGIN \*/\n.*?\n/\* ROUTER_TRACE_END \*/\n', '', generated, flags=re.S)
    if kind == 'cuda':
        body = body[body.index('__launch_bounds__(256, 1)'):body.rindex('}\n')]
        body = body.replace('void route_multi_diagnostic(', 'void route_multi(').replace('float* __restrict__ weights, int n_tok, float* trace)', 'float* __restrict__ weights, int n_tok)')
        return body.replace('    const float trace_full_sum = warp_sum(sum);\n    const float reciprocal = 1.0f / trace_full_sum;', '    const float reciprocal = 1.0f / warp_sum(sum);')
    return body.replace('buffer Status {uint invalid;float trace[26];}', 'buffer Status {uint invalid;}').replace('    precise float trace_full_sum=xor_sum(sum,lane);\n    precise float reciprocal=1.0/trace_full_sum;', '    precise float reciprocal=1.0/xor_sum(sum,lane);')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=PINS)
    parser.add_argument('source', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path)
    args = parser.parse_args()
    source = args.source.read_text()
    generated = generate(args.kind, source)
    expected = cuda_body(source) if args.kind == 'cuda' else source
    if restore(args.kind, generated) != expected:
        raise ValueError('router trace source correspondence failed')
    args.output.write_text(generated)
