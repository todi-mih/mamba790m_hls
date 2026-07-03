#include "mamba_types.h"

void rmsnorm_stage(
    hls::stream<data_t> &in,
    hls::stream<data_t> &out,
    data_t norm_weight[D_MODEL]
){
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=norm_weight cyclic factor=PAR dim=1

    data_t x_local[D_MODEL];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=PAR dim=1

    SEQ_LOOP:
    for (int t = 0; t < SEQ_LEN; t++) {
#pragma HLS PIPELINE off

        READ_LOOP:
        for (int i = 0; i < D_MODEL; i++) {
#pragma HLS PIPELINE II=1
            x_local[i] = in.read();
        }

        acc_t partial[PAR];
#pragma HLS ARRAY_PARTITION variable=partial complete

        INIT_P:
        for (int p = 0; p < PAR; p++) {
#pragma HLS UNROLL
            partial[p] = 0;
        }

        SUMSQ_LOOP:
        for (int i = 0; i < D_MODEL; i += PAR) {
#pragma HLS PIPELINE II=1
            for (int p = 0; p < PAR; p++) {
#pragma HLS UNROLL
                data_t xi = x_local[i+p];
                acc_t sq = (acc_t)xi * (acc_t)xi;
                partial[p] += sq;
            }
        }

        acc_t sum_sq = 0;

        REDUCE_LOOP:
        for (int p = 0; p < PAR; p++) {
#pragma HLS UNROLL
            sum_sq += partial[p];
        }

        float mean_sq = sum_sq.to_float() / (float)D_MODEL + 1e-5f;
        float inv_rms = 1.0f / hls::sqrt(mean_sq);

        SCALE_LOOP:
        for (int i = 0; i < D_MODEL; i++) {
#pragma HLS PIPELINE II=1

            acc_t tmp =
                (acc_t)x_local[i] *
                (acc_t)inv_rms *
                (acc_t)norm_weight[i];

            out.write((data_t)tmp);
        }
    }
}
