#include "mamba_types.h"
#include "lut_functions.h"

typedef ap_fixed<16, 4, AP_TRN, AP_SAT> data_sat_t;

void conv_stage(
    hls::stream<data_t> &in,
    hls::stream<data_t> &out,
    const data_t* W_conv_in,
    const data_t* b_conv_in,
    const data_t* lut_silu_conv_in
){
#pragma HLS INLINE off

    // ── Load arrays locally so the top-level DATAFLOW region never sees
    //    them as shared variables.
    data_t W_conv[D_INNER][D_CONV];
    data_t b_conv[D_INNER];
    int    lut_silu_array[LUT_SIZE];

#pragma HLS BIND_STORAGE variable=W_conv type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=b_conv type=ram_1p impl=bram
#pragma HLS ARRAY_PARTITION variable=W_conv cyclic factor=CHAN_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=W_conv complete              dim=2
#pragma HLS ARRAY_PARTITION variable=b_conv cyclic factor=CHAN_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=lut_silu_array complete dim=1

    for (int i = 0; i < D_INNER; i++)
        for (int k = 0; k < D_CONV; k++)
            W_conv[i][k] = W_conv_in[i * D_CONV + k];

    for (int i = 0; i < D_INNER; i++)
        b_conv[i] = b_conv_in[i];

    for (int i = 0; i < LUT_SIZE; i++)
        lut_silu_array[i] = (int)(int16_t)(uint16_t)lut_silu_conv_in[i].range(15, 0);

    // ── Shift register — NOT static so hardware is idempotent ───────────
    data_t shift_reg[D_INNER][D_CONV];
#pragma HLS ARRAY_PARTITION variable=shift_reg cyclic factor=CHAN_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=shift_reg complete              dim=2

INIT_SHIFT:
    for (int c = 0; c < D_INNER; c++) {
#pragma HLS PIPELINE II=1
        for (int k = 0; k < D_CONV; k++) {
#pragma HLS UNROLL
            shift_reg[c][k] = data_t(0);
        }
    }

SEQ_LOOP:
    for (int t = 0; t < SEQ_LEN; t++) {
CHAN_TILE_LOOP:
        for (int c = 0; c < D_INNER; c += CHAN_PAR) {
            const int tile = ((c + CHAN_PAR) <= D_INNER) ? CHAN_PAR : (D_INNER - c);

            data_t new_samples[CHAN_PAR];
#pragma HLS ARRAY_PARTITION variable=new_samples complete

READ_TILE:
            for (int r = 0; r < CHAN_PAR; r++) {
#pragma HLS UNROLL
                new_samples[r] = (r < tile) ? in.read() : data_t(0);
            }

UPDATE_SHIFT:
            for (int r = 0; r < CHAN_PAR; r++) {
#pragma HLS UNROLL
                if (r < tile) {
                    for (int k = 0; k < D_CONV - 1; k++) {
#pragma HLS UNROLL
                        shift_reg[c + r][k] = shift_reg[c + r][k + 1];
                    }
                    shift_reg[c + r][D_CONV - 1] = new_samples[r];
                }
            }

COMPUTE_TILE:
            for (int r = 0; r < CHAN_PAR; r++) {
#pragma HLS UNROLL
                if (r < tile) {
                    acc_t acc = 0;
                    for (int k = 0; k < D_CONV; k++) {
#pragma HLS UNROLL
                        data_t prod = shift_reg[c + r][k] * W_conv[c + r][k];
#pragma HLS BIND_OP variable=prod op=mul impl=dsp
                        acc += (acc_t)prod;
                    }
                    data_t pre_act = (data_sat_t)(acc + (acc_t)b_conv[c + r]);
                    out.write(lut_silu_fn(pre_act.to_float(), lut_silu_array));
                }
            }
        }
    }
}
