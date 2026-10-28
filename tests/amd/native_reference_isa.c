// Compile inside ggml-cpu, with the same C flags as its Q8 dot implementation.
// Cache options alone do not describe -march=native or user-supplied ISA flags.
const char * strata_amd_native_reference_compiler(void);
const char * strata_amd_native_reference_compiler(void) { return __VERSION__; }
unsigned strata_amd_native_reference_isa(void);
unsigned strata_amd_native_reference_isa(void) {
    unsigned flags = 0;
#ifdef __AVX2__
    flags |= 1u;
#endif
#ifdef __FMA__
    flags |= 2u;
#endif
#ifdef __F16C__
    flags |= 4u;
#endif
#ifdef __AVX512F__
    flags |= 8u;
#endif
#ifdef __AVXVNNI__
    flags |= 16u;
#endif
#ifdef __AVX512VNNI__
    flags |= 32u;
#endif
#ifdef __AVXVNNIINT8__
    flags |= 64u;
#endif
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
    flags |= 128u;
#endif
    return flags;
}
