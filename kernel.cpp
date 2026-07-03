//https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba/modeling_mamba.py?utm_source=chatgpt.com

#include "mamba_types.h"
#include "lut_functions.h"
#include "hls_burst_maxi.h"

// ============================================================
// Stage declarations
// ============================================================
void rmsnorm_stage(hls::stream<data_t>& in, hls::stream<data_t>& out, data_t norm_weight[D_MODEL]);
void gemv_in_proj(hls::stream<data_t>& in, hls::stream<data_t>& x_out, hls::stream<data_t>& z_out, const data_t* W_in_x0, const data_t* W_in_z0);
void conv_stage(hls::stream<data_t>& in, hls::stream<data_t>& out, const data_t* W_conv_in, const data_t* b_conv_in, const data_t* lut_silu_conv_in);
void gemv_x_proj(hls::stream<data_t>& in, hls::stream<data_t>& dt_out, hls::stream<data_t>& B_out, hls::stream<data_t>& C_out, const data_t* W_x_proj);
void gemv_dt_proj(hls::stream<data_t>& in, hls::stream<data_t>& out, const data_t* W_dt_proj_in, const data_t* b_dt_proj_in);
void ssm_stage(hls::stream<data_t>& delta_in, hls::stream<data_t>& B_in, hls::stream<data_t>& C_in, hls::stream<data_t>& x_in, hls::stream<data_t>& out, const data_t* logA_mat_in, const data_t* D_vec_in);
void gating_stage(hls::stream<data_t>& y_in, hls::stream<data_t>& z_in, hls::stream<data_t>& out);
void gemv_out_proj(hls::stream<data_t>& in, hls::stream<data_t>& out, const data_t* W_out_proj_in);

// ============================================================
// HELPERS
// ============================================================

// Load tokens AND copy them to a residual stream for the final add.
// Keeps input AXI port accessed by exactly one DATAFLOW task.
static void load_tokens(
    const data_t* input,
    hls::stream<data_t>& out,
    hls::stream<data_t>& residual)
{
#pragma HLS INLINE off
    for (int t = 0; t < SEQ_LEN; t++)
        for (int i = 0; i < D_MODEL; i++) {
#pragma HLS PIPELINE II=1
            data_t v = input[t * D_MODEL + i];
            out.write(v);
            residual.write(v);
        }
}

static void load_norm_weight(const data_t* src, data_t dst[D_MODEL]){
#pragma HLS INLINE off
    for (int i = 0; i < D_MODEL; i++) dst[i] = src[i];
}

// Split conv output: one to gemv_x_proj, one to ssm_stage x_in
static void tee_conv(
    hls::stream<data_t>& in,
    hls::stream<data_t>& out_xproj,
    hls::stream<data_t>& out_ssm)
{
#pragma HLS INLINE off
    for (int t = 0; t < SEQ_LEN; t++)
        for (int i = 0; i < D_INNER; i++) {
#pragma HLS PIPELINE II=1
            data_t v = in.read();
            out_xproj.write(v);
            out_ssm.write(v);
        }
}

// Store: add residual input to out_proj result, write to output buffer
static void store_with_residual(
    hls::stream<data_t>& proj_in,
    hls::stream<data_t>& residual_in,
    data_t* output)
{
#pragma HLS INLINE off
    int idx = 0;
    for (int t = 0; t < SEQ_LEN; t++)
        for (int i = 0; i < D_MODEL; i++) {
#pragma HLS PIPELINE II=1
            // cast through data_t to keep fixed-point width consistent
            data_t res = (data_t)((acc_t)proj_in.read() + (acc_t)residual_in.read());
            output[idx++] = res;
        }
}

// ============================================================
// TOP
// ============================================================
extern "C" {
void vadd_mul(
    const data_t* input,
    data_t* output,                    // D_MODEL per token (with residual)
    const data_t* W_in_x0,
    const data_t* W_in_z0,
    const data_t* W_x_proj,
    const data_t* W_dt_proj,
    const data_t* W_out0,              // gmem6: out_proj weight
    const data_t* norm_weight_in,
    const data_t* W_conv_in,
    const data_t* b_conv_in,
    const data_t* b_dt_proj_in,
    const data_t* A_mat_in,            // gmem11: unused (kept for interface stability)
    const data_t* logA_mat_in,         // gmem12: log(-A) for ssm
    const data_t* D_vec_in,            // gmem13: skip connection
    const data_t* lut_exp_in,
    const data_t* lut_softplus_in,
    const data_t* lut_silu_conv_in,
    const data_t* lut_silu_gate_in
)
{
#pragma HLS INTERFACE m_axi port=input            bundle=gmem0
#pragma HLS INTERFACE m_axi port=output           bundle=gmem1
#pragma HLS INTERFACE m_axi port=W_in_x0          bundle=gmem2
#pragma HLS INTERFACE m_axi port=W_in_z0          bundle=gmem3
#pragma HLS INTERFACE m_axi port=W_x_proj         bundle=gmem4
#pragma HLS INTERFACE m_axi port=W_dt_proj        bundle=gmem5
#pragma HLS INTERFACE m_axi port=W_out0           bundle=gmem6
#pragma HLS INTERFACE m_axi port=norm_weight_in   bundle=gmem7
#pragma HLS INTERFACE m_axi port=W_conv_in        bundle=gmem8
#pragma HLS INTERFACE m_axi port=b_conv_in        bundle=gmem9
#pragma HLS INTERFACE m_axi port=b_dt_proj_in     bundle=gmem10
#pragma HLS INTERFACE m_axi port=A_mat_in         bundle=gmem11
#pragma HLS INTERFACE m_axi port=logA_mat_in      bundle=gmem12
#pragma HLS INTERFACE m_axi port=D_vec_in         bundle=gmem13
#pragma HLS INTERFACE m_axi port=lut_exp_in       bundle=gmem14
#pragma HLS INTERFACE m_axi port=lut_softplus_in  bundle=gmem15
#pragma HLS INTERFACE m_axi port=lut_silu_conv_in bundle=gmem16
#pragma HLS INTERFACE m_axi port=lut_silu_gate_in bundle=gmem17
#pragma HLS INTERFACE s_axilite port=return bundle=control

    data_t norm_weight[D_MODEL];
#pragma HLS BIND_STORAGE variable=norm_weight type=ram_1p impl=bram
    load_norm_weight(norm_weight_in, norm_weight);

    hls::stream<data_t> s_token_in    ("s_token_in");
    hls::stream<data_t> s_residual    ("s_residual");    // input copy for residual add
    hls::stream<data_t> s_norm_out    ("s_norm_out");
    hls::stream<data_t> s_x_stream    ("s_x_stream");
    hls::stream<data_t> s_z_stream    ("s_z_stream");    // bypass to gating
    hls::stream<data_t> s_conv_out    ("s_conv_out");
    hls::stream<data_t> s_conv_xproj  ("s_conv_xproj");
    hls::stream<data_t> s_conv_ssm    ("s_conv_ssm");
    hls::stream<data_t> s_dt_out      ("s_dt_out");
    hls::stream<data_t> s_B_out       ("s_B_out");
    hls::stream<data_t> s_C_out       ("s_C_out");
    hls::stream<data_t> s_delta_out   ("s_delta_out");
    hls::stream<data_t> s_ssm_out     ("s_ssm_out");
    hls::stream<data_t> s_gated       ("s_gated");
    hls::stream<data_t> s_proj_out    ("s_proj_out");

    // s_residual and s_z_stream carry full SEQ_LEN*D_MODEL /
    // SEQ_LEN*D_INNER worth of data before the consumer starts
#pragma HLS STREAM variable=s_token_in   depth=1024
#pragma HLS STREAM variable=s_residual   depth=49152   // SEQ_LEN*D_MODEL = 32*1536
#pragma HLS STREAM variable=s_norm_out   depth=1024
#pragma HLS STREAM variable=s_x_stream   depth=8192
#pragma HLS STREAM variable=s_z_stream   depth=98304   // SEQ_LEN*D_INNER = 32*3072
#pragma HLS STREAM variable=s_conv_out   depth=8192
#pragma HLS STREAM variable=s_conv_xproj depth=4096
#pragma HLS STREAM variable=s_conv_ssm   depth=4096
#pragma HLS STREAM variable=s_dt_out     depth=1024
#pragma HLS STREAM variable=s_B_out      depth=1024
#pragma HLS STREAM variable=s_C_out      depth=1024
#pragma HLS STREAM variable=s_delta_out  depth=4096
#pragma HLS STREAM variable=s_ssm_out    depth=4096
#pragma HLS STREAM variable=s_gated      depth=4096
#pragma HLS STREAM variable=s_proj_out   depth=1024

#pragma HLS DATAFLOW

    load_tokens    (input,         s_token_in,   s_residual);
    rmsnorm_stage  (s_token_in,    s_norm_out,   norm_weight);
    gemv_in_proj   (s_norm_out,    s_x_stream,   s_z_stream,   W_in_x0,      W_in_z0);
    conv_stage     (s_x_stream,    s_conv_out,   W_conv_in,    b_conv_in,    lut_silu_conv_in);
    tee_conv       (s_conv_out,    s_conv_xproj, s_conv_ssm);
    gemv_x_proj    (s_conv_xproj,  s_dt_out,     s_B_out,      s_C_out,      W_x_proj);
    gemv_dt_proj   (s_dt_out,      s_delta_out,  W_dt_proj,    b_dt_proj_in);
    ssm_stage      (s_delta_out,   s_B_out,      s_C_out,      s_conv_ssm,   s_ssm_out,  logA_mat_in, D_vec_in);
    gating_stage   (s_ssm_out,     s_z_stream,   s_gated);
    gemv_out_proj  (s_gated,       s_proj_out,   W_out0);
    store_with_residual(s_proj_out, s_residual,  output);
}
}
