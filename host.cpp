// host.cpp - Hardware verification + benchmarking: Full Mamba Block
// Loads 5 real tokens, cycles them to fill SEQ_LEN=32 batch.
// Golden-checks tokens 0-4, sanity-checks tokens 5-31.

#include "xrt.h"
#include "experimental/xrt_kernel.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <sys/time.h>
#include <string>
#include <numeric>
#include <algorithm>

// ============================================================
// CONSTANTS
// ============================================================
#define SEQ_LEN   32
#define D_MODEL   1536
#define D_INNER   3072
#define DT_RANK   96
#define D_STATE   16
#define D_CONV    4
#define LUT_SIZE  1024
#define N_GOLDEN  5

#define N_WARMUP  10    // runs discarded before measurement
#define N_BENCH   30    // runs measured 

#define TOL       0.25f // Q4.12 rounding boundary 

#define SILU_LUT_MIN   -10.0f
#define SILU_LUT_MAX    10.0f
#define SILU_LUT_SCALE 256.0f

#define BASE           "/mem_files/"
#define SCALE_NORM     0.02192441187798977f
#define SCALE_INPROJ   0.010219497606158257f
#define SCALE_CONV     0.008677f
#define SCALE_XPROJ    0.005955538712441921f
#define SCALE_DTPROJ   0.006225914694368839f
#define SCALE_OUTPROJ  0.018810033798217773f

using raw_t = uint16_t;

// ============================================================
// TIMING HELPERS
// ============================================================
static inline double now_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

static double vec_mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

static double vec_stddev(const std::vector<double>& v, double mean) {
    double acc = 0.0;
    for (double x : v) acc += (x - mean) * (x - mean);
    return std::sqrt(acc / v.size());
}

static double vec_min(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }
static double vec_max(const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); }

// ============================================================
// POWER / THERMAL HELPERS 
// ============================================================
#define SYSFS_BASE  "/sys/bus/pci/devices/0000:01:00.1/hwmon/"

static std::string find_sysfs_path(const char* filename) {
    for (int i = 0; i < 10; i++) {
        std::string p = std::string(SYSFS_BASE) + "hwmon" + std::to_string(i) + "/" + filename;
        FILE* f = fopen(p.c_str(), "r");
        if (f) { fclose(f); return p; }
    }
    return "";
}

static float read_sysfs_float(const char* filename, float scale) {
    std::string path = find_sysfs_path(filename);
    if (path.empty()) return -1.0f;
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return -1.0f;
    long val;
    int ok = fscanf(f, "%ld", &val);
    fclose(f);
    return (ok == 1) ? (float)val * scale : -1.0f;
}

// Board power in Watts  (sysfs: microwatts)
static float read_power_w() { return read_sysfs_float("power1_average", 1e-6f); }
// FPGA junction temp in Celsius  (sysfs: millidegrees)
static float read_temp_c()  { return read_sysfs_float("temp1_input",    1e-3f); }


// ============================================================
// BUFFERS
// ============================================================
static raw_t input      [SEQ_LEN * D_MODEL];
static raw_t output     [SEQ_LEN * D_MODEL];

static raw_t norm_weight[D_MODEL];
static raw_t W_in_x     [D_INNER * D_MODEL];
static raw_t W_in_z     [D_INNER * D_MODEL];
static raw_t W_conv     [D_INNER * D_CONV];
static raw_t b_conv     [D_INNER];
static raw_t W_x_proj   [(DT_RANK + 2 * D_STATE) * D_INNER];
static raw_t W_dt_proj  [D_INNER * DT_RANK];
static raw_t b_dt_proj  [D_INNER];
static raw_t logA_mat   [D_INNER * D_STATE];
static raw_t D_vec      [D_INNER];
static raw_t W_out_proj [D_MODEL * D_INNER];
static raw_t lut_silu   [LUT_SIZE];
static raw_t dummy_data [65536];

static raw_t all_inputs [N_GOLDEN * D_MODEL];
static raw_t all_golden [N_GOLDEN * D_MODEL];

// ============================================================
// HELPERS
// ============================================================
static inline float to_float(raw_t x) { return ((int16_t)x) / 4096.0f; }

static raw_t float_to_raw(float f) {
    int q = (int)roundf(f * 4096.0f);
    if (q >  32767) q =  32767;
    if (q < -32768) q = -32768;
    return (raw_t)((int16_t)q);
}

static inline float silu_exact(float x) { return x / (1.0f + expf(-x)); }

static void build_lut_silu(raw_t* arr) {
    const float x_range = SILU_LUT_MAX - SILU_LUT_MIN;
    for (int i = 0; i < LUT_SIZE; i++) {
        float x  = SILU_LUT_MIN + (float)i / (float)(LUT_SIZE - 1) * x_range;
        int   sv = (int)roundf(silu_exact(x) * SILU_LUT_SCALE);
        if (sv >  32767) sv =  32767;
        if (sv < -32768) sv = -32768;
        arr[i] = (raw_t)(int16_t)sv;
    }
}

static void load_q412(const char* path, raw_t* arr, int n) {
    FILE* f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    for (int i = 0; i < n; i++) {
        unsigned v;
        if (fscanf(f, "%x", &v) != 1) { printf("EOF in %s at %d\n", path, i); exit(1); }
        arr[i] = (raw_t)(v & 0xFFFF);
    }
    fclose(f);
}

static void load_int8_scaled(const char* path, raw_t* arr, int n, float scale) {
    FILE* f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    for (int i = 0; i < n; i++) {
        unsigned v;
        if (fscanf(f, "%x", &v) != 1) { printf("EOF in %s at %d\n", path, i); exit(1); }
        signed char sv = (signed char)(v & 0xFF);
        arr[i] = float_to_raw((float)sv * scale);
    }
    fclose(f);
}

static xrt::bo make_dummy(xrt::device& dev, xrt::kernel& krnl, int arg_idx) {
    xrt::bo bo(dev, sizeof(dummy_data), krnl.group_id(arg_idx));
    memset(bo.map<void*>(), 0, sizeof(dummy_data));
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return bo;
}

// ============================================================
// VERIFY (called once after warmup to confirm correctness)
// ============================================================
int verify_token(int t, raw_t* got, raw_t* golden) {
    int pass = 0, fail = 0, first_fail = -1;
    float max_err = 0.0f;
    for (int i = 0; i < D_MODEL; i++) {
        float err = fabsf(to_float(got[i]) - to_float(golden[i]));
        if (err > max_err) max_err = err;
        if (err <= TOL) pass++;
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

int sanity_check(int t, raw_t* got) {
    int zeros = 0, nonfinite = 0;
    for (int i = 0; i < D_MODEL; i++) {
        float v = to_float(got[i]);
        if (v == 0.0f)    zeros++;
        if (!std::isfinite(v)) nonfinite++;
    }
    int ok = (nonfinite == 0) && (zeros < D_MODEL / 2);
    printf("  [tok %2d  sanity]  zeros=%d  nonfinite=%d  %s\n",
           t, zeros, nonfinite, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <kernel.xclbin>\n"; return 1; }

    printf("=== Mamba Hardware Benchmark: Full Block (%d-token batch) ===\n", SEQ_LEN);
    printf("    D_MODEL=%d  D_INNER=%d  D_STATE=%d  DT_RANK=%d\n", D_MODEL, D_INNER, D_STATE, DT_RANK);
    printf("    Warmup runs: %d  |  Measured runs: %d  |  Tolerance: %.3f\n\n",
           N_WARMUP, N_BENCH, TOL);

    // ── Load weights ──────────────────────────────────────────────────
    printf("Loading weights...\n");
    load_int8_scaled(BASE "norm_weight.mem",    norm_weight, D_MODEL,                    SCALE_NORM);
    load_int8_scaled(BASE "conv_weight.mem",    W_conv,      D_INNER * D_CONV,            SCALE_CONV);
    load_q412       (BASE "conv_bias.mem",      b_conv,      D_INNER);
    load_int8_scaled(BASE "x_proj_weight.mem",  W_x_proj,    (DT_RANK+2*D_STATE)*D_INNER, SCALE_XPROJ);
    load_int8_scaled(BASE "dt_proj_weight.mem", W_dt_proj,   D_INNER * DT_RANK,           SCALE_DTPROJ);
    load_q412       (BASE "dt_proj_bias.mem",   b_dt_proj,   D_INNER);
    load_q412       (BASE "logA_mat.mem",       logA_mat,    D_INNER * D_STATE);
    load_q412       (BASE "D_vec.mem",          D_vec,       D_INNER);
    load_int8_scaled(BASE "out_proj_weight.mem",W_out_proj,  D_MODEL * D_INNER,           SCALE_OUTPROJ);
    {
        std::vector<raw_t> tmp(D_INNER * 2 * D_MODEL);
        load_int8_scaled(BASE "in_proj_weight.mem", tmp.data(), D_INNER * 2 * D_MODEL, SCALE_INPROJ);
        for (int r = 0; r < D_INNER; r++)
            for (int c = 0; c < D_MODEL; c++) {
                W_in_x[r * D_MODEL + c] = tmp[r             * D_MODEL + c];
                W_in_z[r * D_MODEL + c] = tmp[(D_INNER + r) * D_MODEL + c];
            }
    }
    build_lut_silu(lut_silu);

    printf("Loading %d golden tokens...\n", N_GOLDEN);
    load_q412(BASE "input_5tokens.mem",  all_inputs, N_GOLDEN * D_MODEL);
    load_q412(BASE "golden_5tokens.mem", all_golden, N_GOLDEN * D_MODEL);

    for (int t = 0; t < SEQ_LEN; t++)
        for (int i = 0; i < D_MODEL; i++)
            input[t * D_MODEL + i] = all_inputs[(t % N_GOLDEN) * D_MODEL + i];

    printf("All files loaded.\n\n");

    // ── XRT setup + idle baseline ─────────────────────────────────────
    xrt::device dev(0);

    float idle_power = read_power_w();
    float idle_temp  = read_temp_c();
    printf("Idle baseline  ->  power: %.2f W  |  temp: %.1f C\n\n", idle_power, idle_temp);

    auto uuid = dev.load_xclbin(argv[1]);
    xrt::kernel krnl(dev, uuid, "vadd_mul");

    // ── BO allocation + upload (weights only once) ────────────────────
    auto bo_input    = xrt::bo(dev, sizeof(input),       krnl.group_id(0));
    auto bo_output   = xrt::bo(dev, sizeof(output),      krnl.group_id(1));
    auto bo_win_x    = xrt::bo(dev, sizeof(W_in_x),      krnl.group_id(2));
    auto bo_win_z    = xrt::bo(dev, sizeof(W_in_z),      krnl.group_id(3));
    auto bo_wxproj   = xrt::bo(dev, sizeof(W_x_proj),    krnl.group_id(4));
    auto bo_wdtproj  = xrt::bo(dev, sizeof(W_dt_proj),   krnl.group_id(5));
    auto bo_woutproj = xrt::bo(dev, sizeof(W_out_proj),  krnl.group_id(6));
    auto bo_norm     = xrt::bo(dev, sizeof(norm_weight),  krnl.group_id(7));
    auto bo_conv     = xrt::bo(dev, sizeof(W_conv),       krnl.group_id(8));
    auto bo_bconv    = xrt::bo(dev, sizeof(b_conv),       krnl.group_id(9));
    auto bo_bdtproj  = xrt::bo(dev, sizeof(b_dt_proj),   krnl.group_id(10));
    auto bo_logA     = xrt::bo(dev, sizeof(logA_mat),    krnl.group_id(12));
    auto bo_dvec     = xrt::bo(dev, sizeof(D_vec),        krnl.group_id(13));
    auto bo_lut      = xrt::bo(dev, sizeof(lut_silu),     krnl.group_id(16));
    auto bo_d11 = make_dummy(dev, krnl, 11);
    auto bo_d14 = make_dummy(dev, krnl, 14);
    auto bo_d15 = make_dummy(dev, krnl, 15);
    auto bo_d17 = make_dummy(dev, krnl, 17);

    // Weights: upload once, stay resident for all runs
    memcpy(bo_win_x.map<void*>(),    W_in_x,      sizeof(W_in_x));    bo_win_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bo_win_z.map<void*>(),    W_in_z,      sizeof(W_in_z));    bo_win_z.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bo_wxproj.map<void*>(),   W_x_proj,    sizeof(W_x_proj));  bo_wxproj.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bo_wdtproj.map<void*>(),  W_dt_proj,   sizeof(W_dt_proj)); bo_wdtproj.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bo_woutproj.map<void*>(), W_out_proj,  sizeof(W_out_proj));bo_woutproj.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bo_norm.map<void*>(),     norm_weight,  sizeof(norm_weight));bo_norm.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bo_conv.map<void*>(),     W_conv,       sizeof(W_conv));    bo_conv.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bo_bconv.map<void*>(),    b_conv,       sizeof(b_conv));    bo_bconv.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bo_bdtproj.map<void*>(),  b_dt_proj,    sizeof(b_dt_proj)); bo_bdtproj.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bo_logA.map<void*>(),     logA_mat,     sizeof(logA_mat));  bo_logA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bo_dvec.map<void*>(),     D_vec,        sizeof(D_vec));     bo_dvec.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bo_lut.map<void*>(),      lut_silu,     sizeof(lut_silu));  bo_lut.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Compute H2D weight bytes (reported once, not per-run)
    size_t weight_bytes =
        sizeof(W_in_x) + sizeof(W_in_z)   + sizeof(W_x_proj)  + sizeof(W_dt_proj) +
        sizeof(W_out_proj) + sizeof(norm_weight) + sizeof(W_conv) + sizeof(b_conv)  +
        sizeof(b_dt_proj)  + sizeof(logA_mat)    + sizeof(D_vec)  + sizeof(lut_silu);

    // Helper lambda: one full inference pass, returns (h2d_us, kern_us, d2h_us)
    auto one_run = [&]() -> std::tuple<double,double,double> {
        // H2D: input only (weights already resident)
        memcpy(bo_input.map<void*>(), input, sizeof(input));
        double t0 = now_us();
        bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        double t1 = now_us();

        auto run = krnl(
            bo_input, bo_output,
            bo_win_x, bo_win_z, bo_wxproj, bo_wdtproj, bo_woutproj,
            bo_norm, bo_conv, bo_bconv, bo_bdtproj,
            bo_d11, bo_logA, bo_dvec,
            bo_d14, bo_d15, bo_lut, bo_d17
        );
        run.wait();
        double t2 = now_us();

        bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        double t3 = now_us();
        return {t1-t0, t2-t1, t3-t2};
    };

    // ── Correctness check FIRST (cold SSM state = matches golden) ────
    // Golden was generated with a fresh SSM state (h=0). Must verify
    // before any warmup runs, because each run updates the SSM h-state
    // inside the kernel. After warmup the state has drifted and golden
    // comparison is meaningless (though computation is still correct).
    {
        (void)one_run();
        memcpy(output, bo_output.map<void*>(), sizeof(output));
    }
    printf("=== Correctness (cold SSM state, tol=%.3f) ===\n", TOL);
    int total_fail = 0;
    for (int t = 0; t < SEQ_LEN; t++) {
        raw_t* tok_out = output + t * D_MODEL;
        if (t < N_GOLDEN)
            total_fail += verify_token(t, tok_out, all_golden + t * D_MODEL);
        else
            total_fail += sanity_check(t, tok_out);
    }
    printf("Correctness: %s\n\n", total_fail == 0 ? "PASS" : "FAIL");

    // ── Warmup (after correctness, SSM state drift doesn't matter) ───
    printf("Warming up (%d runs)...\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; i++) {
        one_run();
        printf("  warmup %2d/%-2d done\r", i+1, N_WARMUP);
        fflush(stdout);
    }
    printf("\n");

    // ── Benchmark loop ────────────────────────────────────────────────
    printf("Benchmarking (%d runs)...\n", N_BENCH);
    std::vector<double> h2d_samples, kern_samples, d2h_samples, total_samples;
    h2d_samples.reserve(N_BENCH); kern_samples.reserve(N_BENCH);
    d2h_samples.reserve(N_BENCH); total_samples.reserve(N_BENCH);

    float bench_power = -1.0f, bench_temp = -1.0f;

    for (int i = 0; i < N_BENCH; i++) {
        auto [h2d, kern, d2h] = one_run();
        h2d_samples.push_back(h2d);
        kern_samples.push_back(kern);
        d2h_samples.push_back(d2h);
        total_samples.push_back(h2d + kern + d2h);
        printf("  run %2d/%-2d  kern=%.2f us\r", i+1, N_BENCH, kern);
        fflush(stdout);
    }
    printf("\n");

    // Sample power + temp at end of benchmark (board warm and steady)
    bench_power = read_power_w();
    bench_temp  = read_temp_c();

    // ── Compute statistics ────────────────────────────────────────────
    double kern_mean  = vec_mean(kern_samples);
    double kern_std   = vec_stddev(kern_samples, kern_mean);
    double kern_min   = vec_min(kern_samples);
    double kern_max   = vec_max(kern_samples);
    double total_mean = vec_mean(total_samples);
    double total_std  = vec_stddev(total_samples, total_mean);
    double h2d_mean   = vec_mean(h2d_samples);
    double d2h_mean   = vec_mean(d2h_samples);

    double tokens_per_s_kern  = (double)SEQ_LEN / (kern_mean  * 1e-6);
    double tokens_per_s_e2e   = (double)SEQ_LEN / (total_mean * 1e-6);
    double cv_pct = 100.0 * kern_std / kern_mean;  // coefficient of variation

    // ── Final report ─────────────────────────────────────────────────
    printf("\n");
    printf("=== TIMING  (n=%d, post-%d-run warmup) ===\n", N_BENCH, N_WARMUP);
    printf("  Weight upload   :  one-time  %zu bytes  (%.1f MB)\n",
           weight_bytes, weight_bytes / 1e6);
    printf("  H2D input/run   :  mean %8.2f us   (%zu bytes,  %.1f MB/s)\n",
           h2d_mean, sizeof(input),
           (sizeof(input) / 1e6) / (h2d_mean * 1e-6));
    printf("  Kernel exec     :  mean %8.2f us   stddev %.2f us   CV %.2f%%\n",
           kern_mean, kern_std, cv_pct);
    printf("                     min  %8.2f us   max    %.2f us\n", kern_min, kern_max);
    printf("  D2H output/run  :  mean %8.2f us   (%zu bytes,  %.1f MB/s)\n",
           d2h_mean, sizeof(output),
           (sizeof(output) / 1e6) / (d2h_mean * 1e-6));
    printf("  End-to-end/run  :  mean %8.2f us   stddev %.2f us   (%.3f ms)\n",
           total_mean, total_std, total_mean / 1e3);

    printf("\n=== THROUGHPUT ===\n");
    printf("  Tokens/s  (kernel only)  :  %.2f\n", tokens_per_s_kern);
    printf("  Tokens/s  (end-to-end)   :  %.2f\n", tokens_per_s_e2e);
    printf("  Batch                    :  %d tokens x %d dims\n", SEQ_LEN, D_MODEL);
    printf("  Useful compute           :  %.2f us / token\n", kern_mean / SEQ_LEN);

    printf("\n=== POWER & THERMAL ===\n");
    if (bench_power >= 0.0f) {
        printf("  Idle power          :  %.2f W\n", idle_power);
        printf("  Under-load power    :  %.2f W\n", bench_power);
        if (idle_power >= 0.0f) {
            float delta = bench_power - idle_power;
            printf("  Kernel delta        :  %+.2f W\n", delta);
            if (delta > 0.0f) {
                double energy_uj = delta * (kern_mean * 1e-6) * 1e6;
                printf("  Energy / inference  :  %.2f uJ\n", energy_uj);
                printf("  Efficiency          :  %.2f tokens/s/W\n", tokens_per_s_kern / delta);
            }
        }
    } else {
        printf("  Power data unavailable (check sysfs path)\n");
    }
    if (idle_temp >= 0.0f)   printf("  Idle temp           :  %.1f C\n", idle_temp);
    if (bench_temp >= 0.0f)  printf("  Under-load temp     :  %.1f C\n", bench_temp);
    if (bench_temp > 85.0f)
        printf("  *** WARNING: temp > 85C, clock may be throttled — results unreliable ***\n");
    if (cv_pct > 3.0)
        printf("  *** NOTE: CV=%.1f%% — timing variance is high, check for throttling or background load ***\n", cv_pct);

    printf("\n=== FINAL RESULT: %s ===\n",
           total_fail == 0 ? "PASS" : "FAIL");
    return total_fail == 0 ? 0 : 1;
}