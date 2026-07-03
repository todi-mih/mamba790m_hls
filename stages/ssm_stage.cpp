#include "mamba_types.h"

void ssm_stage(
    hls::stream<data_t>& delta_in,
    hls::stream<data_t>& B_in,
    hls::stream<data_t>& C_in,
    hls::stream<data_t>& x_in,
    hls::stream<data_t>& out,
    const data_t* logA_mat_in,   // [D_INNER * D_STATE] linear A = -exp(A_log), negative, gmem12
    const data_t* D_vec_in       // [D_INNER], skip connection, gmem13
){
#pragma HLS INLINE off

    // ── Load A locally ───────────────────────────────────────────────
    // D_INNER * D_STATE = 3072 * 16 = 49152 elements x 16-bit = 96 KB -> URAM.
    // Values are linear A = -exp(A_log), all negative. Stored as data_t
    // (ap_fixed<16,4>, range [-8, 8)), so any A < -8 is saturated to -8 
    data_t logA_mat[D_INNER][D_STATE];
#pragma HLS BIND_STORAGE variable=logA_mat type=ram_2p impl=uram
#pragma HLS ARRAY_PARTITION variable=logA_mat complete dim=2

    for (int c = 0; c < D_INNER; c++)
        for (int n = 0; n < D_STATE; n++) {
#pragma HLS PIPELINE II=1
            logA_mat[c][n] = logA_mat_in[c * D_STATE + n];
        }

    // ── Load D_vec locally ───────────────────────────────────────────
    data_t D_vec[D_INNER];
#pragma HLS BIND_STORAGE variable=D_vec type=ram_1p impl=bram

    for (int c = 0; c < D_INNER; c++) {
#pragma HLS PIPELINE II=1
        D_vec[c] = D_vec_in[c];
    }

    // ── Persistent hidden state ──────────────────────────────────────
    // Declared static so it survives across tokens, across batches, and
    // across kernel invocations: the state at token t feeds the recurrence
    // at token t+1. As a side effect the kernel is not idempotent between
    // runs, so the host verifies the reference only on the first cold run,
    // when h is zero-initialized.
    static ssm_t h[D_INNER][D_STATE];
#pragma HLS ARRAY_PARTITION variable=h complete dim=2

    data_t B_buf[D_STATE];
    data_t C_buf[D_STATE];
#pragma HLS ARRAY_PARTITION variable=B_buf complete
#pragma HLS ARRAY_PARTITION variable=C_buf complete

    for (int t = 0; t < SEQ_LEN; t++) {

        // ── 1. Read B ────────────────────────────────────────────────
        for (int n = 0; n < D_STATE; n++) {
#pragma HLS PIPELINE II=1
            B_buf[n] = B_in.read();
        }

        // ── 2. Read C ────────────────────────────────────────────────
        for (int n = 0; n < D_STATE; n++) {
#pragma HLS PIPELINE II=1
            C_buf[n] = C_in.read();
        }

        // ── 3. SSM channel loop ──────────────────────────────────────
        // Inner state loop fully unrolled over all D_STATE = 16 components,
        // so the 16 state updates for a channel happen in parallel. The
        // exponential is the dominant arithmetic cost of the whole kernel:
        // 16 unrolled hls::exp calls per channel per token. A LUT-based exp
        // is available in lut_functions.h but the direct float version is
        // kept here for accuracy.
        for (int c = 0; c < D_INNER; c++) {
#pragma HLS PIPELINE
            float delta_c = delta_in.read().to_float();
            float x_c     = x_in.read().to_float();

            ssm_t y_c = 0;

            for (int n = 0; n < D_STATE; n++) {
#pragma HLS UNROLL
                // dA = exp(delta * A): A <= 0 and delta >= 0, so the exponent
                // is <= 0 and the result lies in (0, 1] — no overflow.
                // The exp and the delta-dependent products use float; the
                // state update and accumulation use the wider ssm_t (Q16.16).
                float dA  = hls::exp(delta_c * logA_mat[c][n].to_float());
                float dB  = delta_c * B_buf[n].to_float();

                ssm_t h_new = ssm_t(dA) * h[c][n]
                            + ssm_t(dB) * ssm_t(x_c);
                h[c][n] = h_new;
                y_c += ssm_t(C_buf[n].to_float()) * h_new;
            }

            // Skip connection through D: D[c] * x[c] in ssm_t precision,
            // accumulated into the output, then truncated to Q4.12.
            y_c += ssm_t(D_vec[c].to_float()) * ssm_t(x_c);

            out.write((data_t)y_c);
        }
    }
}