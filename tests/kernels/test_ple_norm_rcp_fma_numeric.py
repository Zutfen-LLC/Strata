"""Independent literal association and candidate-checker refusal controls."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
import compare_ple_norm_rcp_fma as check
fp = check.fp


def refuse(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('invalid candidate accepted')


# Literal frozen before device observation. Reciprocal is an injected F32
# operand; its RNE value is not asserted to equal CUDA MUFU.RCP.
assert fp.fma(0x3f004189, 0x39cccccd, fp.EPS) == 0x394e4218
assert fp.add(fp.nearest(fp.rational(0x3f004189)/2560), fp.EPS) == 0x394e4217
one = 0x3f800000
x, gamma = [one]*10240, [one]*10240
total = check.row_sum(x[:2560])
reciprocal = 0x39cccccd
shifted = fp.fma(total, reciprocal, fp.EPS)
scale = fp.rsqrt_nearest(shifted)
mean = fp.mul(total, reciprocal)
diagnostic = [total, reciprocal, shifted, scale, mean]*4
output = [fp.mul(fp.mul(scale, one), one)]*10240
check.controls(x, gamma, output, diagnostic)
for slot in range(5):
    changed = diagnostic[:]
    # Scale's inherited bound is intentionally unchanged; use an unambiguously
    # invalid value rather than forcing one-ULP exactness for that primitive.
    changed[slot] = 0x40000000 if slot == 4 else (one if slot != 0 else 0)
    refuse(lambda: check.controls(x, gamma, output, changed))
changed = output[:]
changed[0] ^= 1
refuse(lambda: check.controls(x, gamma, changed, diagnostic))
refuse(lambda: check.controls(x[:-1], gamma, output, diagnostic))
print('reciprocal/FMA literal, own-primitive and refusal controls pass; no CUDA equivalence claim')
