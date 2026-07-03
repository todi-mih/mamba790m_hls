#include "mamba_types.h"

void gemv_out_proj(
    hls::stream<data_t>& in,
    hls::stream<data_t>& out,
    const data_t* W_out_proj_in    // [D_MODEL * D_INNER] row-major, AXI (gmem6)
){
#pragma HLS INLINE off

    data_t x_local[D_INNER];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=PAR dim=1

    for (int t = 0; t < SEQ_LEN; t++) {

        // ── 1. Read gated vector (D_INNER) ───────────────────────────
        for (int i = 0; i < D_INNER; i++) {
#pragma HLS PIPELINE II=1
            x_local[i] = in.read();
        }

        // ── 2. GEMV: one row at a time ────────────────────────────────
        for (int row = 0; row < D_MODEL; row++) {
            acc_t acc = 0;
            for (int j = 0; j < D_INNER; j++) {
#pragma HLS PIPELINE
                acc += (acc_t)W_out_proj_in[row * D_INNER + j] * (acc_t)x_local[j];
            }
            out.write((data_t)acc);
        }
    }
}
