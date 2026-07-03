#include "mamba_types.h"

void gating_stage(
    hls::stream<data_t>& y_in,    // from ssm_stage
    hls::stream<data_t>& z_in,    // from gemv_in_proj bypass
    hls::stream<data_t>& out
){
#pragma HLS INLINE off

    for (int t = 0; t < SEQ_LEN; t++) {
        for (int c = 0; c < D_INNER; c++) {
#pragma HLS PIPELINE II=1
            float y   = y_in.read().to_float();
            float z   = z_in.read().to_float();
            float sig = 1.0f / (1.0f + hls::exp(-z));
            out.write(data_t(y * (z * sig)));
        }
    }
}
