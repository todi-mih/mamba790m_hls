#include "mamba_types.h"

void gemv_x_proj(
    hls::stream<data_t>& in,
    hls::stream<data_t>& dt_out,
    hls::stream<data_t>& B_out,
    hls::stream<data_t>& C_out,
    const data_t* W_x_proj
){
#pragma HLS INLINE off

    const int OUT_ROWS = DT_RANK + 2 * D_STATE;  // 96 + 32 = 128

    data_t x_local[D_INNER];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=PAR dim=1

    for (int t = 0; t < SEQ_LEN; t++) {

        // -- 1. Read one token from conv_stage --------------------------
        for (int i = 0; i < D_INNER; i++) {
#pragma HLS PIPELINE II=1
            x_local[i] = in.read();
        }

        // -- 2. GEMV: one output row at a time --------------------------
        for (int row = 0; row < OUT_ROWS; row++) {
            acc_t acc = 0;
            for (int j = 0; j < D_INNER; j++) {
#pragma HLS PIPELINE
                acc += (acc_t)W_x_proj[row * D_INNER + j] * (acc_t)x_local[j];
            }

            data_t val = (data_t)acc;

            // -- 3. Route to correct output stream ----------------------
            if (row < DT_RANK)
                dt_out.write(val);
            else if (row < DT_RANK + D_STATE)
                B_out.write(val);
            else
                C_out.write(val);
        }
    }
}