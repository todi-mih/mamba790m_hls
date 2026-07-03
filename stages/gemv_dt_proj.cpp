#include "mamba_types.h"

void gemv_dt_proj(
    hls::stream<data_t>& in,
    hls::stream<data_t>& out,
    const data_t* W_dt_proj_in,   // [D_INNER * DT_RANK]
    const data_t* b_dt_proj_in    // [D_INNER]
){
#pragma HLS INLINE off

    // ── Load bias locally — D_INNER=3072 × 16-bit = 6 KB, fits in BRAM ─
    data_t b_local[D_INNER];
#pragma HLS BIND_STORAGE variable=b_local type=ram_1p impl=bram

    for (int i = 0; i < D_INNER; i++) {
#pragma HLS PIPELINE II=1
        b_local[i] = b_dt_proj_in[i];
    }

    // ── dt buffer for one token ───────────────────────────────────────
    data_t dt_local[DT_RANK];
#pragma HLS ARRAY_PARTITION variable=dt_local cyclic factor=PAR dim=1

    for (int t = 0; t < SEQ_LEN; t++) {

        // ── 1. Read DT_RANK values from gemv_x_proj ──────────────────
        for (int i = 0; i < DT_RANK; i++) {
#pragma HLS PIPELINE II=1
            dt_local[i] = in.read();
        }

        // ── 2. GEMV + bias + softplus, one row at a time ─────────────
        for (int row = 0; row < D_INNER; row++) {
            acc_t acc = 0;
            for (int j = 0; j < DT_RANK; j++) {
#pragma HLS PIPELINE
                acc += (acc_t)W_dt_proj_in[row * DT_RANK + j] * (acc_t)dt_local[j];
            }

            // bias
            float xf = (acc + (acc_t)b_local[row]).to_float();

            // softplus: log(1 + exp(x)), numerically stable
            float sp;
            if      (xf >  20.0f) sp = xf;
            else if (xf < -20.0f) sp = 0.0f;
            else                  sp = hls::log(1.0f + hls::exp(xf));

            out.write(data_t(sp));
        }
    }
}
