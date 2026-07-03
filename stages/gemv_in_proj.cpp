#include "mamba_types.h"

void gemv_in_proj(
    hls::stream<data_t> &in,
    hls::stream<data_t> &x_out,
    hls::stream<data_t> &z_out,
    const data_t        *W_in_x0,
    const data_t        *W_in_z0
){
#pragma HLS INLINE off

    // ── Local input buffer ──────────────────────────────
    data_t x_local[D_MODEL];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=PAR dim=1

    // ── Weight tile buffers (ROW_PAR rows × D_MODEL cols) ───────────
    // Declared outside the tile loop so HLS can infer BRAM.
    data_t w_x[ROW_PAR][D_MODEL];
    data_t w_z[ROW_PAR][D_MODEL];
#pragma HLS BIND_STORAGE variable=w_x type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=w_z type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=w_x cyclic factor=PAR dim=2
#pragma HLS ARRAY_PARTITION variable=w_z cyclic factor=PAR dim=2

    acc_t acc_x[ROW_PAR];
    acc_t acc_z[ROW_PAR];
#pragma HLS ARRAY_PARTITION variable=acc_x complete
#pragma HLS ARRAY_PARTITION variable=acc_z complete

TOKEN_LOOP:
    for (int t = 0; t < SEQ_LEN; t++) {
#pragma HLS PIPELINE off   // token loop must not be pipelined across iterations
                        // since x_local needs to be reused

    READ_INPUT:
        for (int i = 0; i < D_MODEL; i++) {
#pragma HLS PIPELINE II=1
            x_local[i] = in.read();
        }

    ROW_TILE_LOOP:
        for (int row = 0; row < D_INNER; row += ROW_PAR) {

            // ── Load ROW_PAR rows of W_in_x0 (burst per row) ────────
        LOAD_WX:
            for (int r = 0; r < ROW_PAR; r++) {
                for (int j = 0; j < D_MODEL; j++) {
#pragma HLS PIPELINE II=1
                    w_x[r][j] = W_in_x0[(row + r) * D_MODEL + j];
                }
            }

            // ── Load ROW_PAR rows of W_in_z0 (burst per row) ────────
        LOAD_WZ:
            for (int r = 0; r < ROW_PAR; r++) {
                for (int j = 0; j < D_MODEL; j++) {
#pragma HLS PIPELINE II=1
                    w_z[r][j] = W_in_z0[(row + r) * D_MODEL + j];
                }
            }

            // ── Zero accumulators ────────────────────────────────────
        INIT_ACC:
            for (int r = 0; r < ROW_PAR; r++) {
#pragma HLS UNROLL
                acc_x[r] = 0;
                acc_z[r] = 0;
            }

            // ── PAR-wide MAC, ROW_PAR rows simultaneously ────────────
        DOT_LOOP:
            for (int j = 0; j < D_MODEL; j += PAR) {
#pragma HLS PIPELINE II=1
                for (int p = 0; p < PAR; p++) {
#pragma HLS UNROLL
                    data_t x = x_local[j + p];
                    for (int r = 0; r < ROW_PAR; r++) {
#pragma HLS UNROLL
                        acc_x[r] += (acc_t)w_x[r][j + p] * (acc_t)x;
                        acc_z[r] += (acc_t)w_z[r][j + p] * (acc_t)x;
                    }
                }
            }

            // ── Emit ROW_PAR results to x_out and z_out ─────────────
            // Sequential (no UNROLL) so each stream gets one write/cycle.
            // x_out and z_out are independent FIFOs so both can accept
            // a write in the same cycle even without UNROLL.
        WRITE_OUT:
            for (int r = 0; r < ROW_PAR; r++) {
                x_out.write((data_t)acc_x[r]);
                z_out.write((data_t)acc_z[r]);
            }
        }
    }
}
