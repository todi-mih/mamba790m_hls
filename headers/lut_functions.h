#ifndef LUT_FUNCTIONS_H
#define LUT_FUNCTIONS_H

#include "mamba_types.h"

// ── Shared LUT definitions ────────────────────────────────────────────
// All LUTs: 1024 entries, scale=256 (divide output by 256 to get float)
// exp LUT:      input range [-20,  0]
// softplus LUT: input range [  0, 20]
// silu LUT:     input range [-10, 10]

#define LUT_SIZE     1024
#define LUT_SCALE    256.0f

#define LUT_SCALE_INV (1.0f / 256.0f)

// exp: x in [-20, 0]
#define EXP_MIN     -20.0f
#define EXP_MAX      0.0f
#define EXP_RANGE    20.0f

// softplus: x in [0, 20]
#define SPLUS_MIN    0.0f
#define SPLUS_MAX    20.0f
#define SPLUS_RANGE  20.0f

// silu: x in [-10, 10]
#define SILU_MIN    -10.0f
#define SILU_MAX     10.0f
#define SILU_RANGE   20.0f

// ── LUT index helper ──────────────────────────────────────────────────
// Isolated so HLS can see the index computation as a clean integer path
// and schedule it independently from the table read.
static inline int lut_index(float x, float x_min, float range) {
#pragma HLS INLINE
    int idx = (int)((x - x_min) / range * (float)(LUT_SIZE - 1));
    if (idx < 0)         idx = 0;
    if (idx >= LUT_SIZE) idx = LUT_SIZE - 1;
    return idx;
}

static inline data_t lut_exp_fn(float x, int lut[LUT_SIZE]) {
#pragma HLS INLINE
    if (x >= EXP_MAX)  return data_t(1.0f);
    if (x <= EXP_MIN)  return data_t(0.0f);
    int idx = lut_index(x, EXP_MIN, EXP_RANGE);
    return data_t((float)lut[idx] * LUT_SCALE_INV);
}

static inline data_t lut_softplus_fn(float x, int lut[LUT_SIZE]) {
#pragma HLS INLINE
    if (x >= SPLUS_MAX) return data_t(x);
    if (x <= SPLUS_MIN) return data_t(0.0f);
    int idx = lut_index(x, SPLUS_MIN, SPLUS_RANGE);
    return data_t((float)lut[idx] * LUT_SCALE_INV);
}

static inline data_t lut_silu_fn(float x, int lut[LUT_SIZE]) {
#pragma HLS INLINE
    if (x >= SILU_MAX)  return data_t(x);
    if (x <= SILU_MIN)  return data_t(0.0f);
    int idx = lut_index(x, SILU_MIN, SILU_RANGE);
    return data_t((float)lut[idx] * LUT_SCALE_INV);
}

#endif
