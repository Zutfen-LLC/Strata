#ifndef STRATA_NATIVE_HC_MIX_LIMITS_H
#define STRATA_NATIVE_HC_MIX_LIMITS_H
// Largest positive F32 satisfying the documented uniform exp/denominator proof.
#define STRATA_HC_MIX_ACTIVATION_LIMIT_BITS 0x42aeac4cu
// Four minimum-normal ULPs reserve division/threshold-rounding headroom.
#define STRATA_HC_MIX_MIN_QUOTIENT_BITS 0x00800004u
#define STRATA_HC_MIX_MAX_DENOMINATOR_BITS 0x7e800000u
#endif
