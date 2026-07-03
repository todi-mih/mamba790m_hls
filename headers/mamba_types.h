#ifndef MAMBA_TYPES_H
#define MAMBA_TYPES_H

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>

typedef ap_fixed<16,4>   data_t;   // activations [-8,8)

typedef ap_fixed<40,20>  acc_t;    // GEMV accumulators, wide enough for D_MODEL=1536 terms

typedef ap_fixed<32,16>  ssm_t;
typedef ap_fixed<16,6>   logA_t; 

#define D_MODEL        1536
#define D_INNER        3072
#define D_STATE        16
#define D_CONV         4
#define DT_RANK        96
#define SEQ_LEN        32

// ── Parallelism knobs ──────────────────────────────────────────────────
#define PAR            4
#define ROW_PAR        2
#define CHAN_PAR       2

// ── LUT metadata
#define LUT_SIZE       1024
#define SILU_LUT_SCALE 256.0f
#define SILU_LUT_MIN  -10.0f
#define SILU_LUT_MAX   10.0f

#endif
