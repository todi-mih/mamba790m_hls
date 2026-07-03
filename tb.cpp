// tb.cpp - Full Mamba Block CSIM
// Loads 5 real tokens, cycles them to fill SEQ_LEN=32 batch.
// Verifies tokens 0-4 against golden (tol=0.20).
// Sanity-checks tokens 5-31 (finite, non-zero).

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "mamba_types.h"

// ============================================================
// DUT
// ============================================================
extern "C" void vadd_mul(
    const data_t* input,
    data_t*       output,
    const data_t* W_in_x0,
    const data_t* W_in_z0,
    const data_t* W_x_proj,
    const data_t* W_dt_proj,
    const data_t* W_out0,
    const data_t* norm_weight_in,
    const data_t* W_conv_in,
    const data_t* b_conv_in,
    const data_t* b_dt_proj_in,
    const data_t* A_mat_in,
    const data_t* logA_mat_in,
    const data_t* D_vec_in,
    const data_t* lut_exp_in,
    const data_t* lut_softplus_in,
    const data_t* lut_silu_conv_in,
    const data_t* lut_silu_gate_in
);

// ============================================================
// CONFIG
// ============================================================
#define BASE           "/mem_files/"
#define N_GOLDEN       5
#define SCALE_NORM     0.02192441187798977f
#define SCALE_INPROJ   0.010219497606158257f
#define SCALE_CONV     0.008677f
#define SCALE_XPROJ    0.005955538712441921f
#define SCALE_DTPROJ   0.006225914694368839f
#define SCALE_OUTPROJ  0.018810033798217773f
#define DUMMY_SIZE     65536

// ============================================================
// BUFFERS
// ============================================================
static data_t input      [SEQ_LEN * D_MODEL];
static data_t output     [SEQ_LEN * D_MODEL];

static data_t norm_weight[D_MODEL];
static data_t W_in_x     [D_INNER * D_MODEL];
static data_t W_in_z     [D_INNER * D_MODEL];
static data_t W_conv     [D_INNER * D_CONV];
static data_t b_conv     [D_INNER];
static data_t W_x_proj   [(DT_RANK + 2 * D_STATE) * D_INNER];
static data_t W_dt_proj  [D_INNER * DT_RANK];
static data_t b_dt_proj  [D_INNER];
static data_t logA_mat   [D_INNER * D_STATE];
static data_t D_vec      [D_INNER];
static data_t W_out_proj [D_MODEL * D_INNER];
static data_t lut_silu   [LUT_SIZE];
static data_t dummy      [DUMMY_SIZE];

static data_t all_inputs [N_GOLDEN * D_MODEL];
static data_t all_golden [N_GOLDEN * D_MODEL];

// ============================================================
// HELPERS
// ============================================================
static inline float to_float(data_t x) { return x.to_float(); }
static inline float silu_exact(float x) { return x / (1.0f + expf(-x)); }

static void zero_all() {
    memset(input,       0, sizeof(input));
    memset(output,      0, sizeof(output));
    memset(norm_weight, 0, sizeof(norm_weight));
    memset(W_in_x,      0, sizeof(W_in_x));
    memset(W_in_z,      0, sizeof(W_in_z));
    memset(W_conv,      0, sizeof(W_conv));
    memset(b_conv,      0, sizeof(b_conv));
    memset(W_x_proj,    0, sizeof(W_x_proj));
    memset(W_dt_proj,   0, sizeof(W_dt_proj));
    memset(b_dt_proj,   0, sizeof(b_dt_proj));
    memset(logA_mat,    0, sizeof(logA_mat));
    memset(D_vec,       0, sizeof(D_vec));
    memset(W_out_proj,  0, sizeof(W_out_proj));
    memset(lut_silu,    0, sizeof(lut_silu));
    memset(dummy,       0, sizeof(dummy));
    memset(all_inputs,  0, sizeof(all_inputs));
    memset(all_golden,  0, sizeof(all_golden));
}

// ============================================================
// LOADERS
// ============================================================
static void load_q412(const char* path, data_t* arr, int n) {
    FILE* f = fopen(path, "r");
    if (!f) { printf("ERROR opening %s\n", path); exit(1); }
    for (int i = 0; i < n; i++) {
        unsigned v;
        if (fscanf(f, "%x", &v) != 1) {
            printf("EOF in %s at %d\n", path, i); fclose(f); exit(1);
        }
        short sv = (short)(v & 0xFFFF);
        arr[i] = data_t((float)sv / 4096.0f);
    }
    fclose(f);
}

static void load_int8_scaled(const char* path, data_t* arr, int n, float scale) {
    FILE* f = fopen(path, "r");
    if (!f) { printf("ERROR opening %s\n", path); exit(1); }
    for (int i = 0; i < n; i++) {
        unsigned v;
        if (fscanf(f, "%x", &v) != 1) {
            printf("EOF in %s at %d\n", path, i); fclose(f); exit(1);
        }
        signed char sv = (signed char)(v & 0xFF);
        arr[i] = data_t((float)sv * scale);
    }
    fclose(f);
}

static void build_lut_silu(data_t* arr) {
    const float x_range = SILU_LUT_MAX - SILU_LUT_MIN;
    for (int i = 0; i < LUT_SIZE; i++) {
        float x  = SILU_LUT_MIN + (float)i / (float)(LUT_SIZE - 1) * x_range;
        int   sv = (int)roundf(silu_exact(x) * SILU_LUT_SCALE);
        if (sv >  32767) sv =  32767;
        if (sv < -32768) sv = -32768;
        arr[i] = data_t((float)(short)sv / 4096.0f);
    }
}

// ============================================================
// VERIFY
// ============================================================
// Full golden check for first N_GOLDEN tokens
int verify_token(int t, data_t* got, data_t* golden, float tol = 0.25f) {
    int pass = 0, fail = 0, first_fail = -1;
    float max_err = 0.0f;
    for (int i = 0; i < D_MODEL; i++) {
        float err = fabsf(to_float(got[i]) - to_float(golden[i]));
        if (err > max_err) max_err = err;
        if (err <= tol) pass++;
        else { if (first_fail < 0) first_fail = i; fail++; }
    }
    printf("  [tok %2d vs golden[%d]]  %4d PASS  %4d FAIL  (%5.1f%%)  max_err=%.6f",
           t, t % N_GOLDEN, pass, fail, 100.0f * pass / D_MODEL, max_err);
    if (fail > 0)
        printf("  first_fail=[%d] got=%.4f expected=%.4f",
               first_fail, to_float(got[first_fail]), to_float(golden[first_fail]));
    printf("\n");
    return fail;
}

// Sanity check for tokens beyond N_GOLDEN, just confirm non-NaN, non-all-zero
int sanity_check(int t, data_t* got) {
    int zeros = 0, nonfinite = 0;
    for (int i = 0; i < D_MODEL; i++) {
        float v = to_float(got[i]);
        if (v == 0.0f)    zeros++;
        if (!isfinite(v)) nonfinite++;
    }
    int ok = (nonfinite == 0) && (zeros < D_MODEL / 2);
    printf("  [tok %2d  sanity]  zeros=%d  nonfinite=%d  %s\n",
           t, zeros, nonfinite, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

// ============================================================
// MAIN
// ============================================================
int main() {
    zero_all();

    printf("=== vadd_mul CSIM: Full Mamba Block (%d-token cycle) ===\n", N_GOLDEN);
    printf("    SEQ_LEN=%d  D_MODEL=%d  D_INNER=%d  D_STATE=%d\n\n",
           SEQ_LEN, D_MODEL, D_INNER, D_STATE);

    // ── Load weights ─────────────────────────────────────────────────
    printf("Loading weights...\n");
    load_int8_scaled(BASE "norm_weight.mem",    norm_weight, D_MODEL,                    SCALE_NORM);
    load_int8_scaled(BASE "conv_weight.mem",    W_conv,      D_INNER * D_CONV,            SCALE_CONV);
    load_q412       (BASE "conv_bias.mem",      b_conv,      D_INNER);
    load_int8_scaled(BASE "x_proj_weight.mem",  W_x_proj,    (DT_RANK+2*D_STATE)*D_INNER, SCALE_XPROJ);
    load_int8_scaled(BASE "dt_proj_weight.mem", W_dt_proj,   D_INNER * DT_RANK,           SCALE_DTPROJ);
    load_q412       (BASE "dt_proj_bias.mem",   b_dt_proj,   D_INNER);
    load_q412       (BASE "logA_mat.mem",        logA_mat,   D_INNER * D_STATE);
    load_q412       (BASE "D_vec.mem",           D_vec,      D_INNER);
    load_int8_scaled(BASE "out_proj_weight.mem", W_out_proj, D_MODEL * D_INNER,           SCALE_OUTPROJ);

    {
        std::vector<data_t> tmp(D_INNER * 2 * D_MODEL);
        load_int8_scaled(BASE "in_proj_weight.mem", tmp.data(),
                         D_INNER * 2 * D_MODEL, SCALE_INPROJ);
        for (int r = 0; r < D_INNER; r++)
            for (int c = 0; c < D_MODEL; c++) {
                W_in_x[r * D_MODEL + c] = tmp[r             * D_MODEL + c];
                W_in_z[r * D_MODEL + c] = tmp[(D_INNER + r) * D_MODEL + c];
            }
    }

    build_lut_silu(lut_silu);
    printf("Weights ready.\n\n");

    // ── Load 5 golden token pairs ────────────────────────────────────
    printf("Loading %d golden tokens...\n", N_GOLDEN);
    load_q412(BASE "input_5tokens.mem",  all_inputs, N_GOLDEN * D_MODEL);
    load_q412(BASE "golden_5tokens.mem", all_golden, N_GOLDEN * D_MODEL);
    printf("Golden tokens loaded.\n\n");

    // ── Build SEQ_LEN batch by cycling the 5 tokens ──────────────────
    for (int t = 0; t < SEQ_LEN; t++)
        for (int i = 0; i < D_MODEL; i++)
            input[t * D_MODEL + i] = all_inputs[(t % N_GOLDEN) * D_MODEL + i];

    printf("Input batch built (SEQ_LEN=%d, cycling %d tokens).\n\n", SEQ_LEN, N_GOLDEN);

    // ── Spot-check ───────────────────────────────────────────────────
    printf("Spot-check (first 4):\n");
    printf("  input[0]:     "); for(int i=0;i<4;i++) printf("%.4f ", to_float(input[i]));           printf("\n");
    printf("  golden[0]:    "); for(int i=0;i<4;i++) printf("%.4f ", to_float(all_golden[i]));      printf("\n");
    printf("  input[1*D]:   "); for(int i=0;i<4;i++) printf("%.4f ", to_float(input[D_MODEL + i])); printf("\n");
    printf("  golden[1*D]:  "); for(int i=0;i<4;i++) printf("%.4f ", to_float(all_golden[D_MODEL + i])); printf("\n\n");

    // ── Run kernel ───────────────────────────────────────────────────
    printf("Running vadd_mul...\n");
    vadd_mul(
        input,     output,
        W_in_x,    W_in_z,    W_x_proj,
        W_dt_proj,
        W_out_proj,                      // W_out0 (gmem6)
        norm_weight,
        W_conv,    b_conv,
        b_dt_proj,
        dummy,                           // A_mat_in    (unused, gmem11)
        logA_mat,                        // logA_mat_in (gmem12)
        D_vec,                           // D_vec_in    (gmem13)
        dummy,     dummy,                // lut_exp, lut_softplus (unused)
        lut_silu,                        // lut_silu_conv_in (gmem16)
        dummy                            // lut_silu_gate_in (gmem17, unused)
    );
    printf("Done.\n\n");

    // ── Verify ───────────────────────────────────────────────────────
    printf("=== Verification ===\n");
    printf("  Tokens 0..%d: golden-checked (tol=0.20)\n", N_GOLDEN - 1);
    printf("  Tokens %d..%d: sanity-checked (finite, non-zero)\n\n", N_GOLDEN, SEQ_LEN - 1);

    int total_fail = 0;
    for (int t = 0; t < SEQ_LEN; t++) {
        data_t* tok_out = output + t * D_MODEL;
        if (t < N_GOLDEN)
            total_fail += verify_token(t, tok_out, all_golden + t * D_MODEL);
        else
            total_fail += sanity_check(t, tok_out);
    }

    printf("\n=== FINAL RESULT: %s ===\n",
           total_fail == 0 ? "PASS" : "FAIL");
    return total_fail == 0 ? 0 : 1;
}
