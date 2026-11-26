// Exact C++20 implementation of the unconstrained 27-block two-block Transformer (Definition 2)
// and causal shortcut-calibration intervention experiments (Natural, Suppressed c=0,
// Pre-calibrated c=c*, and Suppressed->Restored), with finite-difference gradient verification.
//
// Compile on macOS (Apple Accelerate BLAS + multithreading):
//   make -C cpp check

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include "cblas_shim.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct Config {
  int n = 8;
  int sigma = 4;
  int d_model = 64;
  int n_heads = 3;
  int d_k = 24;
  int d_mlp = 128;
  double init_scale = 0.02;
  double lr = 1e-3;
  double weight_decay = 0.01;
  int batch_size = 256;
  int sym_quads = 32; // 32 * 4 = 128 antithetic balanced probe sequences
  int total_steps = 2400;
  int eval_every = 100;
  int release_step = 1000;
  int test_size = 1000;
  int n_threads = 8;
};

// Mode of causal intervention on the balanced state-only logit component \bar c_t(\theta)
enum class Mode {
  NATURAL = 0,     // Unmodified 27-block Transformer training
  SUPPRESSED = 1,  // Project balanced state-only logit component \bar c_t -> 0
  CALIBRATED = 2,  // Hold balanced state-only logit component \bar c_t -> c* = log((1-p)/p)
  RESTORED = 3     // Suppress \bar c_t -> 0 for steps < release_step, then release/set c*
};

struct Prompt {
  std::vector<int> tokens; // length L = 2*n + 2
  std::vector<int> states; // s_0, s_1, ..., s_n (length n + 1)
  std::vector<int> bits;   // b_1, ..., b_n (length n)
  int w_star = 0;
};

// All 27 parameter blocks + 6 relative lag bias vectors (Block 1 & 2, or Block 2)
struct Params {
  int vocab_size; // 2 states (0,1) + sigma letters (2..2+sigma-1) + 1 SEP (2+sigma)
  int seq_len;    // 2*n + 2
  int d_model, n_heads, d_k, d_mlp;

  // Flat storage of all parameters so optimizers and finite-diff checks iterate cleanly
  std::vector<double> data;
  struct BlockSlice {
    std::string name;
    size_t offset;
    size_t size;
    bool is_lag_bias;
    bool no_wd;
  };
  std::vector<BlockSlice> blocks;

  // Offsets for fast indexing
  size_t off_E_tok, off_E_pos;
  size_t off_WQ[2][3], off_WK[2][3], off_WV[2][3], off_beta[2][3];
  size_t off_Wup[2], off_bup[2], off_Wdown[2];
  size_t off_V;

  void allocate(const Config& cfg) {
    vocab_size = 3 + cfg.sigma;
    seq_len = 2 * cfg.n + 2;
    d_model = cfg.d_model;
    n_heads = cfg.n_heads;
    d_k = cfg.d_k;
    d_mlp = cfg.d_mlp;
    blocks.clear();
    size_t cur = 0;
    auto add = [&](const std::string& nm, size_t sz, bool is_beta = false, bool no_wd = false) {
      size_t o = cur;
      blocks.push_back({nm, o, sz, is_beta, no_wd});
      cur += sz;
      return o;
    };
    off_E_tok = add("E_tok", (size_t)vocab_size * d_model);
    off_E_pos = add("E_pos", (size_t)seq_len * d_model);
    for (int l = 0; l < 2; ++l) {
      for (int h = 0; h < n_heads; ++h) {
        std::string pfx = "B" + std::to_string(l + 1) + "H" + std::to_string(h + 1) + "_";
        off_WQ[l][h] = add(pfx + "WQ", (size_t)d_k * d_model);
        off_WK[l][h] = add(pfx + "WK", (size_t)d_k * d_model);
        off_WV[l][h] = add(pfx + "WV", (size_t)d_model * d_model);
      }
      std::string mpfx = "B" + std::to_string(l + 1) + "_";
      off_Wup[l] = add(mpfx + "Wup", (size_t)d_mlp * d_model);
      off_bup[l] = add(mpfx + "bup", (size_t)d_mlp, false, true);
      off_Wdown[l] = add(mpfx + "Wdown", (size_t)d_model * d_mlp);
    }
    off_V = add("V", (size_t)2 * d_model);
    // Block-2 relative lag biases \beta^{(2,h)} (and Block-1 fixed 0 or trained; appendix line 1367 uses Block-2 \beta^{(2,h)})
    for (int l = 0; l < 2; ++l) {
      for (int h = 0; h < n_heads; ++h) {
        std::string pfx = "B" + std::to_string(l + 1) + "H" + std::to_string(h + 1) + "_beta";
        off_beta[l][h] = add(pfx, (size_t)seq_len, true, true);
      }
    }
    data.assign(cur, 0.0);
  }

  void init_random(uint64_t seed, double scale, bool zero_V = true) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, scale);
    for (const auto& b : blocks) {
      if (b.is_lag_bias || (zero_V && b.name == "V")) {
        std::fill(data.begin() + b.offset, data.begin() + b.offset + b.size, 0.0);
      } else {
        for (size_t i = 0; i < b.size; ++i) {
          data[b.offset + i] = nd(rng);
        }
      }
    }
  }
};

// Per-sequence forward activation cache for exact analytical backpropagation
struct SeqCache {
  int L, d_model, d_k, d_mlp;
  std::vector<int> tokens;
  std::vector<double> h0;          // L * d_model
  std::vector<double> q[2][3];     // L * d_k
  std::vector<double> k[2][3];     // L * d_k
  std::vector<double> v[2][3];     // L * d_model
  std::vector<double> alpha[2][3]; // L * L (causal lower-triangular)
  std::vector<double> htilde[2];   // L * d_model
  std::vector<double> u_mlp[2];    // L * d_mlp
  std::vector<double> r_mlp[2];    // L * d_mlp
  std::vector<double> h_out[2];    // L * d_model
  std::vector<double> logits;      // L * 2

  void init(const Config& cfg) {
    L = 2 * cfg.n + 2;
    d_model = cfg.d_model;
    d_k = cfg.d_k;
    d_mlp = cfg.d_mlp;
    h0.resize(L * d_model);
    for (int l = 0; l < 2; ++l) {
      for (int h = 0; h < 3; ++h) {
        q[l][h].resize(L * d_k);
        k[l][h].resize(L * d_k);
        v[l][h].resize(L * d_model);
        alpha[l][h].assign(L * L, 0.0);
      }
      htilde[l].resize(L * d_model);
      u_mlp[l].resize(L * d_mlp);
      r_mlp[l].resize(L * d_mlp);
      h_out[l].resize(L * d_model);
    }
    logits.resize(L * 2);
  }
};

// C = alpha * A * B^T + beta * C  (A: M x K, B: N x K, C: M x N, row-major)
inline void gemm_nt(int M, int N, int K, double alpha, const double* A, const double* B, double beta, double* C) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, N);
}
// C = alpha * A * B + beta * C  (A: M x K, B: K x N, C: M x N, row-major)
inline void gemm_nn(int M, int N, int K, double alpha, const double* A, const double* B, double beta, double* C) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
}
// C = alpha * A^T * B + beta * C  (A: K x M, B: K x N, C: M x N, row-major)
inline void gemm_tn(int M, int N, int K, double alpha, const double* A, const double* B, double beta, double* C) {
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C, N);
}

void forward_seq(const Params& P, const std::vector<int>& tokens, SeqCache& C, int upto_pos = -1) {
  const int L = (upto_pos > 0) ? upto_pos : P.seq_len;
  const int D = P.d_model;
  const int DK = P.d_k;
  const int DM = P.d_mlp;
  const double inv_sqrt_dk = 1.0 / std::sqrt((double)DK);

  C.tokens = tokens;
  const double* E_tok = P.data.data() + P.off_E_tok;
  const double* E_pos = P.data.data() + P.off_E_pos;
  for (int i = 0; i < L; ++i) {
    const double* et = E_tok + tokens[i] * D;
    const double* ep = E_pos + i * D;
    double* dst = C.h0.data() + i * D;
    for (int d = 0; d < D; ++d) dst[d] = et[d] + ep[d];
  }

  double scores[64]; // L <= 64
  for (int l = 0; l < 2; ++l) {
    const double* hin = (l == 0) ? C.h0.data() : C.h_out[0].data();
    std::memcpy(C.htilde[l].data(), hin, sizeof(double) * L * D);

    for (int h = 0; h < 3; ++h) {
      const double* WQ = P.data.data() + P.off_WQ[l][h];
      const double* WK = P.data.data() + P.off_WK[l][h];
      const double* WV = P.data.data() + P.off_WV[l][h];
      const double* beta = P.data.data() + P.off_beta[l][h];

      // q = hin * WQ^T (L x DK), k = hin * WK^T (L x DK), v = hin * WV^T (L x D)
      gemm_nt(L, DK, D, 1.0, hin, WQ, 0.0, C.q[l][h].data());
      gemm_nt(L, DK, D, 1.0, hin, WK, 0.0, C.k[l][h].data());
      gemm_nt(L, D, D, 1.0, hin, WV, 0.0, C.v[l][h].data());

      // Causal attention scores and softmax
      double* A = C.alpha[l][h].data();
      for (int i = 0; i < L; ++i) {
        const double* qi = C.q[l][h].data() + i * DK;
        double max_s = -1e300;
        for (int j = 0; j <= i; ++j) {
          const double* kj = C.k[l][h].data() + j * DK;
          double dot = 0.0;
          for (int d = 0; d < DK; ++d) dot += qi[d] * kj[d];
          int lag = i - j;
          // Block-2 relative lag biases active (lag 0 fixed to 0); Block-1 lag bias kept 0 matching Appendix E.3
          double b_lag = (l == 1 && lag > 0) ? beta[lag] : 0.0;
          double s = b_lag + dot * inv_sqrt_dk;
          scores[j] = s;
          if (s > max_s) max_s = s;
        }
        double sum_exp = 0.0;
        for (int j = 0; j <= i; ++j) {
          double e = std::exp(scores[j] - max_s);
          A[i * P.seq_len + j] = e;
          sum_exp += e;
        }
        double inv_sum = 1.0 / sum_exp;
        for (int j = 0; j <= i; ++j) A[i * P.seq_len + j] *= inv_sum;
        for (int j = i + 1; j < L; ++j) A[i * P.seq_len + j] = 0.0;

        // Accumulate attention output into htilde[l][i]
        double* hti = C.htilde[l].data() + i * D;
        for (int j = 0; j <= i; ++j) {
          double w = A[i * P.seq_len + j];
          const double* vj = C.v[l][h].data() + j * D;
          for (int d = 0; d < D; ++d) hti[d] += w * vj[d];
        }
      }
    }

    // ReLU MLP: u = htilde * Wup^T + bup; r = max(0, u); h_out = htilde + r * Wdown^T
    const double* Wup = P.data.data() + P.off_Wup[l];
    const double* bup = P.data.data() + P.off_bup[l];
    const double* Wdown = P.data.data() + P.off_Wdown[l];
    gemm_nt(L, DM, D, 1.0, C.htilde[l].data(), Wup, 0.0, C.u_mlp[l].data());
    for (int i = 0; i < L; ++i) {
      double* ui = C.u_mlp[l].data() + i * DM;
      double* ri = C.r_mlp[l].data() + i * DM;
      for (int m = 0; m < DM; ++m) {
        double val = ui[m] + bup[m];
        ui[m] = val;
        ri[m] = (val > 0.0) ? val : 0.0;
      }
    }
    std::memcpy(C.h_out[l].data(), C.htilde[l].data(), sizeof(double) * L * D);
    gemm_nt(L, D, DM, 1.0, C.r_mlp[l].data(), Wdown, 1.0, C.h_out[l].data());
  }

  // Final linear readout: logits = h_out[1] * V^T (L x 2)
  const double* V = P.data.data() + P.off_V;
  gemm_nt(L, 2, D, 1.0, C.h_out[1].data(), V, 0.0, C.logits.data());
}

// Backward pass for a single sequence given d_logits (L x 2), accumulating into grad_data
struct BackwardWorkspace {
  std::vector<double> dh_out[2], dhtilde[2], du_mlp[2], dr_mlp[2];
  std::vector<double> dq, dk, dv, dA, dS;
  void init(const Config& cfg) {
    int L = 2 * cfg.n + 2;
    for (int l = 0; l < 2; ++l) {
      dh_out[l].resize(L * cfg.d_model);
      dhtilde[l].resize(L * cfg.d_model);
      du_mlp[l].resize(L * cfg.d_mlp);
      dr_mlp[l].resize(L * cfg.d_mlp);
    }
    dq.resize(L * cfg.d_k);
    dk.resize(L * cfg.d_k);
    dv.resize(L * cfg.d_model);
    dA.resize(L * L);
    dS.resize(L * L);
  }
};

void backward_seq(const Params& P, const SeqCache& C, const double* d_logits,
                  BackwardWorkspace& W, double* grad) {
  const int L = P.seq_len;
  const int D = P.d_model;
  const int DK = P.d_k;
  const int DM = P.d_mlp;
  const double inv_sqrt_dk = 1.0 / std::sqrt((double)DK);

  // logits = h_out[1] * V^T  (V: 2 x D)
  // dV += d_logits^T * h_out[1] (2 x D); dh_out[1] = d_logits * V (L x D)
  const double* V = P.data.data() + P.off_V;
  gemm_tn(2, D, L, 1.0, d_logits, C.h_out[1].data(), 1.0, grad + P.off_V);
  gemm_nn(L, D, 2, 1.0, d_logits, V, 0.0, W.dh_out[1].data());

  for (int l = 1; l >= 0; --l) {
    const double* hin = (l == 0) ? C.h0.data() : C.h_out[0].data();
    const double* Wup = P.data.data() + P.off_Wup[l];
    const double* Wdown = P.data.data() + P.off_Wdown[l];

    // h_out[l] = htilde[l] + r_mlp[l] * Wdown^T  (Wdown: D x DM)
    std::memcpy(W.dhtilde[l].data(), W.dh_out[l].data(), sizeof(double) * L * D);
    // dWdown += dh_out[l]^T * r_mlp[l] (D x DM)
    gemm_tn(D, DM, L, 1.0, W.dh_out[l].data(), C.r_mlp[l].data(), 1.0, grad + P.off_Wdown[l]);
    // dr_mlp = dh_out[l] * Wdown (L x DM)
    gemm_nn(L, DM, D, 1.0, W.dh_out[l].data(), Wdown, 0.0, W.dr_mlp[l].data());

    double* dbup = grad + P.off_bup[l];
    for (int i = 0; i < L; ++i) {
      const double* ui = C.u_mlp[l].data() + i * DM;
      const double* dri = W.dr_mlp[l].data() + i * DM;
      double* dui = W.du_mlp[l].data() + i * DM;
      for (int m = 0; m < DM; ++m) {
        double g = (ui[m] > 0.0) ? dri[m] : 0.0;
        dui[m] = g;
        dbup[m] += g;
      }
    }
    // u_mlp = htilde[l] * Wup^T + bup  (Wup: DM x D)
    gemm_tn(DM, D, L, 1.0, W.du_mlp[l].data(), C.htilde[l].data(), 1.0, grad + P.off_Wup[l]);
    gemm_nn(L, D, DM, 1.0, W.du_mlp[l].data(), Wup, 1.0, W.dhtilde[l].data());

    // Attention backward: htilde[l] = hin + sum_h A[l][h] * v[l][h]
    double* dhin = (l == 1) ? W.dh_out[0].data() : W.dh_out[1].data(); // reuse dh_out[1] as dh0 when l==0
    std::memcpy(dhin, W.dhtilde[l].data(), sizeof(double) * L * D);

    for (int h = 0; h < 3; ++h) {
      const double* WQ = P.data.data() + P.off_WQ[l][h];
      const double* WK = P.data.data() + P.off_WK[l][h];
      const double* WV = P.data.data() + P.off_WV[l][h];
      const double* A = C.alpha[l][h].data();
      double* dbeta = grad + P.off_beta[l][h];

      std::fill(W.dv.begin(), W.dv.end(), 0.0);
      std::fill(W.dq.begin(), W.dq.end(), 0.0);
      std::fill(W.dk.begin(), W.dk.end(), 0.0);

      for (int i = 0; i < L; ++i) {
        const double* dhti = W.dhtilde[l].data() + i * D;
        double dot_adA = 0.0;
        double dAi[64];
        for (int j = 0; j <= i; ++j) {
          const double* vj = C.v[l][h].data() + j * D;
          double da = 0.0;
          for (int d = 0; d < D; ++d) da += dhti[d] * vj[d];
          dAi[j] = da;
          dot_adA += A[i * L + j] * da;

          double aij = A[i * L + j];
          double* dvj = W.dv.data() + j * D;
          for (int d = 0; d < D; ++d) dvj[d] += aij * dhti[d];
        }
        const double* qi = C.q[l][h].data() + i * DK;
        double* dqi = W.dq.data() + i * DK;
        for (int j = 0; j <= i; ++j) {
          double ds = A[i * L + j] * (dAi[j] - dot_adA);
          int lag = i - j;
          if (l == 1 && lag > 0) dbeta[lag] += ds;
          double ds_scaled = ds * inv_sqrt_dk;
          const double* kj = C.k[l][h].data() + j * DK;
          double* dkj = W.dk.data() + j * DK;
          for (int d = 0; d < DK; ++d) {
            dqi[d] += ds_scaled * kj[d];
            dkj[d] += ds_scaled * qi[d];
          }
        }
      }

      // Accumulate into dWQ, dWK, dWV and dhin
      gemm_tn(DK, D, L, 1.0, W.dq.data(), hin, 1.0, grad + P.off_WQ[l][h]);
      gemm_tn(DK, D, L, 1.0, W.dk.data(), hin, 1.0, grad + P.off_WK[l][h]);
      gemm_tn(D, D, L, 1.0, W.dv.data(), hin, 1.0, grad + P.off_WV[l][h]);

      gemm_nn(L, D, DK, 1.0, W.dq.data(), WQ, 1.0, dhin);
      gemm_nn(L, D, DK, 1.0, W.dk.data(), WK, 1.0, dhin);
      gemm_nn(L, D, D, 1.0, W.dv.data(), WV, 1.0, dhin);
    }
  }

  // Embedding backward from dh0 (stored in W.dh_out[1])
  const double* dh0 = W.dh_out[1].data();
  double* dE_tok = grad + P.off_E_tok;
  double* dE_pos = grad + P.off_E_pos;
  for (int i = 0; i < L; ++i) {
    const double* g = dh0 + i * D;
    double* dt = dE_tok + C.tokens[i] * D;
    double* dp = dE_pos + i * D;
    for (int d = 0; d < D; ++d) {
      dt[d] += g[d];
      dp[d] += g[d];
    }
  }
}

Prompt sample_prompt(std::mt19937_64& rng, int n, int sigma, double p) {
  Prompt pr;
  pr.tokens.resize(2 * n + 2);
  pr.states.resize(n + 1);
  pr.bits.resize(n);
  std::uniform_int_distribution<int> bit_dist(0, 1);
  std::uniform_int_distribution<int> q_dist(0, sigma - 1);
  std::uniform_int_distribution<int> nonmatch_dist(0, sigma - 2);
  std::uniform_real_distribution<double> u_dist(0.0, 1.0);

  pr.states[0] = bit_dist(rng);
  pr.w_star = q_dist(rng);
  pr.tokens[0] = pr.states[0];
  pr.tokens[1] = 2 + pr.w_star;
  for (int t = 1; t <= n; ++t) {
    bool match = (u_dist(rng) < p);
    int wt;
    if (match) {
      wt = pr.w_star;
    } else {
      int r = nonmatch_dist(rng);
      wt = (r >= pr.w_star) ? (r + 1) : r;
    }
    int bt = match ? 1 : 0;
    pr.bits[t - 1] = bt;
    pr.states[t] = (pr.states[t - 1] + bt) & 1;
    pr.tokens[1 + t] = 2 + wt;
  }
  pr.tokens[n + 2] = 2 + sigma; // SEP
  for (int t = 2; t <= n; ++t) {
    pr.tokens[n + 1 + t] = pr.states[t - 1];
  }
  return pr;
}

// Antithetic 4-tuple under p = 1/2: {(s0=0, b), (s0=1, b), (s0=0, 1-b), (s0=1, 1-b)}
void sample_antithetic_probe(std::mt19937_64& rng, int n, int sigma, int n_quads, std::vector<Prompt>& out) {
  out.clear();
  out.reserve(n_quads * 4);
  std::uniform_int_distribution<int> bit_dist(0, 1);
  std::uniform_int_distribution<int> q_dist(0, sigma - 1);
  std::uniform_int_distribution<int> nonmatch_dist(0, sigma - 2);

  for (int q = 0; q < n_quads; ++q) {
    int w_star = q_dist(rng);
    std::vector<int> base_bits(n);
    std::vector<int> match_tok(n), nonmatch_tok(n);
    for (int t = 0; t < n; ++t) {
      base_bits[t] = bit_dist(rng);
      match_tok[t] = w_star;
      int r = nonmatch_dist(rng);
      nonmatch_tok[t] = (r >= w_star) ? (r + 1) : r;
    }
    for (int flip_b = 0; flip_b < 2; ++flip_b) {
      for (int s0 = 0; s0 < 2; ++s0) {
        Prompt pr;
        pr.tokens.resize(2 * n + 2);
        pr.states.resize(n + 1);
        pr.bits.resize(n);
        pr.w_star = w_star;
        pr.states[0] = s0;
        pr.tokens[0] = s0;
        pr.tokens[1] = 2 + w_star;
        for (int t = 1; t <= n; ++t) {
          int bt = flip_b ? (1 - base_bits[t - 1]) : base_bits[t - 1];
          pr.bits[t - 1] = bt;
          pr.states[t] = (pr.states[t - 1] + bt) & 1;
          pr.tokens[1 + t] = 2 + (bt ? match_tok[t - 1] : nonmatch_tok[t - 1]);
        }
        pr.tokens[n + 2] = 2 + sigma;
        for (int t = 2; t <= n; ++t) pr.tokens[n + 1 + t] = pr.states[t - 1];
        out.push_back(std::move(pr));
      }
    }
  }
}

// Compute intervened batch loss and exact parameter gradients (multithreaded)
double compute_batch_loss_and_grad(
    const Config& cfg,
    const Params& P,
    const std::vector<Prompt>& train_batch,
    const std::vector<Prompt>& sym_batch,
    bool project_shortcut,
    double c_target,
    std::vector<double>& grad_out,
    std::vector<SeqCache>& train_caches,
    std::vector<SeqCache>& sym_caches,
    std::vector<BackwardWorkspace>& bws) {

  const int B = (int)train_batch.size();
  const int Bsym = (int)sym_batch.size();
  const int n = cfg.n;
  const int L = P.seq_len;
  const int NT = std::max(1, cfg.n_threads);

  // 1. Parallel forward pass on train_batch and sym_batch (if project_shortcut)
  auto run_fwd = [&](const std::vector<Prompt>& batch, std::vector<SeqCache>& caches) {
    int N = (int)batch.size();
    std::vector<std::thread> threads;
    for (int tid = 0; tid < NT; ++tid) {
      threads.emplace_back([&, tid]() {
        for (int m = tid; m < N; m += NT) {
          forward_seq(P, batch[m].tokens, caches[m]);
        }
      });
    }
    for (auto& th : threads) th.join();
  };

  run_fwd(train_batch, train_caches);
  std::vector<double> c_bar(n + 1, 0.0); // 1-indexed t = 1..n
  if (project_shortcut) {
    run_fwd(sym_batch, sym_caches);
    for (int t = 1; t <= n; ++t) {
      int pos = n + 1 + t;
      double acc = 0.0;
      for (int m = 0; m < Bsym; ++m) {
        double x_prev = (sym_batch[m].states[t - 1] == 0) ? 1.0 : -1.0;
        double dz = sym_caches[m].logits[pos * 2 + 0] - sym_caches[m].logits[pos * 2 + 1];
        acc += x_prev * dz;
      }
      c_bar[t] = acc / Bsym;
    }
  }

  // 2. Compute loss and per-sequence logit gradients on train_batch
  std::vector<double> train_dlogits(B * L * 2, 0.0);
  std::vector<double> sym_coeff(n + 1, 0.0);
  double total_loss = 0.0;
  const double inv_Bn = 1.0 / ((double)B * n);

  for (int m = 0; m < B; ++m) {
    for (int t = 1; t <= n; ++t) {
      int pos = n + 1 + t;
      double x_prev = (train_batch[m].states[t - 1] == 0) ? 1.0 : -1.0;
      double z0 = train_caches[m].logits[pos * 2 + 0];
      double z1 = train_caches[m].logits[pos * 2 + 1];
      if (project_shortcut) {
        double shift = 0.5 * (c_target - c_bar[t]) * x_prev;
        z0 += shift;
        z1 -= shift;
      }
      double mx = std::max(z0, z1);
      double e0 = std::exp(z0 - mx);
      double e1 = std::exp(z1 - mx);
      double p0 = e0 / (e0 + e1);
      double p1 = e1 / (e0 + e1);
      int y = train_batch[m].states[t];
      total_loss += -(y == 0 ? std::log(p0 + 1e-300) : std::log(p1 + 1e-300)) * inv_Bn;

      double g0 = (p0 - (y == 0 ? 1.0 : 0.0)) * inv_Bn;
      double g1 = (p1 - (y == 1 ? 1.0 : 0.0)) * inv_Bn;
      train_dlogits[(m * L + pos) * 2 + 0] = g0;
      train_dlogits[(m * L + pos) * 2 + 1] = g1;

      if (project_shortcut) {
        // d/d(c_bar[t]) of (z0_int * g0 + z1_int * g1) = -0.5 * x_prev * (g0 - g1)
        sym_coeff[t] += -0.5 * x_prev * (g0 - g1);
      }
    }
  }

  std::vector<double> sym_dlogits;
  if (project_shortcut) {
    sym_dlogits.assign(Bsym * L * 2, 0.0);
    for (int m = 0; m < Bsym; ++m) {
      for (int t = 1; t <= n; ++t) {
        int pos = n + 1 + t;
        double x_prev = (sym_batch[m].states[t - 1] == 0) ? 1.0 : -1.0;
        double coeff = sym_coeff[t] * (x_prev / Bsym);
        sym_dlogits[(m * L + pos) * 2 + 0] = +coeff;
        sym_dlogits[(m * L + pos) * 2 + 1] = -coeff;
      }
    }
  }

  // 3. Parallel backward pass across threads with thread-local gradient buffers
  std::vector<std::vector<double>> thread_grads(NT, std::vector<double>(P.data.size(), 0.0));
  {
    std::vector<std::thread> threads;
    for (int tid = 0; tid < NT; ++tid) {
      threads.emplace_back([&, tid]() {
        double* gbuf = thread_grads[tid].data();
        for (int m = tid; m < B; m += NT) {
          backward_seq(P, train_caches[m], train_dlogits.data() + m * L * 2, bws[tid], gbuf);
        }
        if (project_shortcut) {
          for (int m = tid; m < Bsym; m += NT) {
            backward_seq(P, sym_caches[m], sym_dlogits.data() + m * L * 2, bws[tid], gbuf);
          }
        }
      });
    }
    for (auto& th : threads) th.join();
  }

  std::fill(grad_out.begin(), grad_out.end(), 0.0);
  for (int tid = 0; tid < NT; ++tid) {
    for (size_t i = 0; i < grad_out.size(); ++i) grad_out[i] += thread_grads[tid][i];
  }
  return total_loss;
}

// Verify analytical gradients against centered finite differences across all parameter blocks
bool verify_gradients() {
  Config cfg;
  cfg.n = 4;
  cfg.sigma = 4;
  cfg.d_model = 12;
  cfg.d_k = 6;
  cfg.d_mlp = 16;
  cfg.batch_size = 8;
  cfg.sym_quads = 2;
  cfg.n_threads = 2;

  Params P;
  P.allocate(cfg);
  P.init_random(12345, 0.35, false);
  // Shift MLP biases away from 0 by +-0.5 so eps=1e-6 never crosses a non-differentiable ReLU kink
  for (int l = 0; l < 2; ++l) {
    for (int m = 0; m < cfg.d_mlp; ++m) {
      double& b = P.data[P.off_bup[l] + m];
      b = (b >= 0.0) ? (b + 0.5) : (b - 0.5);
    }
  }
  // Also populate nonzero lag biases so beta gradients are checked thoroughly
  std::mt19937_64 rng(999);
  std::normal_distribution<double> nd(0.0, 0.05);
  for (int h = 0; h < 3; ++h) {
    for (int lag = 1; lag < P.seq_len; ++lag) P.data[P.off_beta[1][h] + lag] = nd(rng);
  }

  std::vector<Prompt> tr(cfg.batch_size), sym;
  for (int m = 0; m < cfg.batch_size; ++m) tr[m] = sample_prompt(rng, cfg.n, cfg.sigma, 0.25);
  sample_antithetic_probe(rng, cfg.n, cfg.sigma, cfg.sym_quads, sym);

  std::vector<SeqCache> tc(cfg.batch_size), sc(sym.size());
  for (auto& c : tc) c.init(cfg);
  for (auto& c : sc) c.init(cfg);
  std::vector<BackwardWorkspace> bws(cfg.n_threads);
  for (auto& w : bws) w.init(cfg);

  for (int mode_chk = 0; mode_chk < 2; ++mode_chk) {
    bool proj = (mode_chk == 1);
    std::vector<double> grad(P.data.size(), 0.0), dummy_grad(P.data.size(), 0.0);
    compute_batch_loss_and_grad(cfg, P, tr, sym, proj, 1.0986, grad, tc, sc, bws);

    double max_rel = 0.0;
    const double eps = 1e-6;
    // Check 3 random entries in every single trainable block
    for (const auto& b : P.blocks) {
      if (b.name.rfind("B1", 0) == 0 && b.is_lag_bias) continue; // B1 lag bias unused by design
      for (int k = 0; k < 3; ++k) {
        size_t idx = b.offset + ((k * 7 + 1) % b.size);
        if (b.is_lag_bias && (idx - b.offset) == 0) idx = b.offset + 1;
        Params P_plus = P, P_minus = P;
        P_plus.data[idx] += eps;
        P_minus.data[idx] -= eps;
        double lp = compute_batch_loss_and_grad(cfg, P_plus, tr, sym, proj, 1.0986, dummy_grad, tc, sc, bws);
        double lm = compute_batch_loss_and_grad(cfg, P_minus, tr, sym, proj, 1.0986, dummy_grad, tc, sc, bws);
        double fd = (lp - lm) / (2.0 * eps);
        double an = grad[idx];
        double denom = std::max({std::abs(fd), std::abs(an), 1e-8});
        double rel = std::abs(fd - an) / denom;
        if (rel > 1e-6) {
          std::cout << "  MISMATCH block=" << b.name << " idx=" << (idx - b.offset)
                    << " fd=" << fd << " an=" << an << " rel=" << rel << "\n";
        }
        if (rel > max_rel) max_rel = rel;
      }
    }
    std::cout << "[GradCheck proj=" << proj << "] max_rel_err = " << std::scientific << max_rel << "\n";
    if (max_rel > 1e-5) return false;
  }
  return true;
}

struct EvalStats {
  double step_loss = 0.0;
  double one_step_acc = 0.0;
  double rollout_acc = 0.0;       // Exact full-sequence continuation accuracy (%)
  double c_sym = 0.0;             // Balanced state-only logit component \bar c
  double lag_n_share = 0.0;       // Max-head within-letter attention share at true lag \delta* = n in Block 2
  double lag_n_mean_heads = 0.0;  // Mean within-letter share at \delta* = n across Block 2 heads
  double E_dist = 0.0;            // Mean |wrong-lag residual correlation|
  double E_true = 0.0;            // Mean true-lag residual correlation
};

EvalStats evaluate_model(
    const Config& cfg,
    const Params& P,
    const std::vector<Prompt>& test_set,
    const std::vector<Prompt>& eval_sym,
    bool project_shortcut,
    double c_target) {

  const int N = (int)test_set.size();
  const int Bsym = (int)eval_sym.size();
  const int n = cfg.n;
  const int L = P.seq_len;

  SeqCache C;
  C.init(cfg);
  std::vector<double> c_bar(n + 1, 0.0);
  for (int m = 0; m < Bsym; ++m) {
    forward_seq(P, eval_sym[m].tokens, C);
    for (int t = 1; t <= n; ++t) {
      int pos = n + 1 + t;
      double x_prev = (eval_sym[m].states[t - 1] == 0) ? 1.0 : -1.0;
      double dz = C.logits[pos * 2 + 0] - C.logits[pos * 2 + 1];
      c_bar[t] += x_prev * dz / Bsym;
    }
  }

  EvalStats st;
  double c_sum = 0.0;
  for (int t = 2; t <= n; ++t) c_sum += (project_shortcut ? c_target : c_bar[t]);
  st.c_sym = c_sum / (n - 1);

  double head_lag_sum[3] = {0.0, 0.0, 0.0};
  std::vector<std::vector<double>> corr_mat(n + 1, std::vector<double>(n + 1, 0.0));

  int exact_rollouts = 0;
  long long one_step_ok = 0;

  for (int m = 0; m < N; ++m) {
    const Prompt& pr = test_set[m];
    forward_seq(P, pr.tokens, C);

    for (int t = 1; t <= n; ++t) {
      int pos = n + 1 + t;
      double x_prev = (pr.states[t - 1] == 0) ? 1.0 : -1.0;
      double z0 = C.logits[pos * 2 + 0];
      double z1 = C.logits[pos * 2 + 1];
      if (project_shortcut) {
        double shift = 0.5 * (c_target - c_bar[t]) * x_prev;
        z0 += shift;
        z1 -= shift;
      }
      double mx = std::max(z0, z1);
      double e0 = std::exp(z0 - mx), e1 = std::exp(z1 - mx);
      double p0 = e0 / (e0 + e1), p1 = e1 / (e0 + e1);
      int y = pr.states[t];
      st.step_loss += -(y == 0 ? std::log(p0 + 1e-300) : std::log(p1 + 1e-300)) / ((double)N * n);
      int pred = (z0 >= z1) ? 0 : 1;
      if (pred == y) one_step_ok++;

      if (t >= 2) {
        // Residual r_hat = 2 * x_{t-1} * (1[s_t == 0] - p0), which equals \chi_t \sigma(-M_t) on the manifold!
        double r_signed = 2.0 * x_prev * ((y == 0 ? 1.0 : 0.0) - p0);
        for (int r = 1; r <= n; ++r) {
          double chi_r = pr.bits[r - 1] ? -1.0 : 1.0;
          corr_mat[t][r] += (r_signed * chi_r) / N;
        }
        // Within-letter attention share at true lag j_t = 1 + t (lag = n) in Block 2
        for (int h = 0; h < 3; ++h) {
          const double* A = C.alpha[1][h].data() + pos * L;
          double let_sum = 0.0;
          for (int r = 1; r <= n; ++r) let_sum += A[1 + r];
          double pi_star = (let_sum > 1e-12) ? (A[1 + t] / let_sum) : (1.0 / n);
          head_lag_sum[h] += pi_star / ((double)N * (n - 1));
        }
      }
    }

    // Free-running autoregressive greedy rollout from prompt (positions 0..n+2)
    std::vector<int> roll_toks = pr.tokens;
    bool all_ok = true;
    for (int t = 1; t <= n; ++t) {
      int pos = n + 1 + t;
      forward_seq(P, roll_toks, C, pos + 1);
      int prev_s = (t == 1) ? pr.states[0] : roll_toks[pos];
      double x_prev = (prev_s == 0) ? 1.0 : -1.0;
      double z0 = C.logits[pos * 2 + 0];
      double z1 = C.logits[pos * 2 + 1];
      if (project_shortcut) {
        double shift = 0.5 * (c_target - c_bar[t]) * x_prev;
        z0 += shift;
        z1 -= shift;
      }
      int pred_s = (z0 >= z1) ? 0 : 1;
      if (pred_s != pr.states[t]) all_ok = false;
      if (t < n) roll_toks[pos + 1] = pred_s;
    }
    if (all_ok) exact_rollouts++;
  }

  st.one_step_acc = 100.0 * (double)one_step_ok / ((double)N * n);
  st.rollout_acc = 100.0 * (double)exact_rollouts / (double)N;
  st.lag_n_share = std::max({head_lag_sum[0], head_lag_sum[1], head_lag_sum[2]});
  st.lag_n_mean_heads = (head_lag_sum[0] + head_lag_sum[1] + head_lag_sum[2]) / 3.0;

  double ed_sum = 0.0, et_sum = 0.0;
  for (int t = 2; t <= n; ++t) {
    et_sum += corr_mat[t][t];
    for (int r = 1; r <= n; ++r) {
      if (r != t) ed_sum += std::abs(corr_mat[t][r]);
    }
  }
  st.E_true = et_sum / (n - 1);
  st.E_dist = ed_sum / ((double)(n - 1) * (n - 1));
  return st;
}

int main(int argc, char** argv) {
  if (!verify_gradients()) {
    std::cerr << "Gradient verification FAILED!\n";
    return 1;
  }
  std::cout << "All 27 blocks + lag biases + shortcut projection passed finite-difference verification!\n";

  if (argc > 1 && std::string(argv[1]) == "--check-only") return 0;

  Config cfg;
  double p = 0.25;
  int seed = 42;
  int mode_int = 0;
  std::string csv_path = "";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--p" && i + 1 < argc) p = std::stod(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = std::stoi(argv[++i]);
    else if (a == "--mode" && i + 1 < argc) mode_int = std::stoi(argv[++i]);
    else if (a == "--steps" && i + 1 < argc) cfg.total_steps = std::stoi(argv[++i]);
    else if (a == "--release" && i + 1 < argc) cfg.release_step = std::stoi(argv[++i]);
    else if (a == "--eval-every" && i + 1 < argc) cfg.eval_every = std::stoi(argv[++i]);
    else if (a == "--csv" && i + 1 < argc) csv_path = argv[++i];
  }

  Mode mode = static_cast<Mode>(mode_int);
  const double c_star = std::log((1.0 - p) / p);

  Params P;
  P.allocate(cfg);
  P.init_random(seed, cfg.init_scale, true);

  // Fixed held-out test set and fixed evaluation balanced probe
  std::mt19937_64 test_rng(100000 + seed);
  std::vector<Prompt> test_set(cfg.test_size);
  for (int m = 0; m < cfg.test_size; ++m) test_set[m] = sample_prompt(test_rng, cfg.n, cfg.sigma, p);
  std::vector<Prompt> eval_sym;
  sample_antithetic_probe(test_rng, cfg.n, cfg.sigma, 64, eval_sym); // 256 balanced probe prompts

  // Paired data RNG so Natural, Suppressed, Calibrated, and Restored see IDENTICAL minibatches in IDENTICAL order!
  std::mt19937_64 data_rng(200000 + seed);
  std::mt19937_64 sym_rng(300000 + seed);

  // AdamW optimizer buffers
  std::vector<double> m_buf(P.data.size(), 0.0), v_buf(P.data.size(), 0.0), grad(P.data.size(), 0.0);
  const double b1 = 0.9, b2 = 0.999, eps_adam = 1e-8;

  std::vector<SeqCache> train_caches(cfg.batch_size), sym_caches(cfg.sym_quads * 4);
  for (auto& c : train_caches) c.init(cfg);
  for (auto& c : sym_caches) c.init(cfg);
  std::vector<BackwardWorkspace> bws(cfg.n_threads);
  for (auto& w : bws) w.init(cfg);

  std::ofstream ofs;
  if (!csv_path.empty()) {
    ofs.open(csv_path);
    ofs << "step,loss,loss_bits,one_step_acc,rollout_acc,c_sym,c_ratio,lag_n_share,E_dist,E_true\n";
  }

  for (int step = 0; step <= cfg.total_steps; ++step) {
    bool proj = false;
    double c_targ = 0.0;
    if (mode == Mode::SUPPRESSED) {
      proj = true;
      c_targ = 0.0;
    } else if (mode == Mode::CALIBRATED) {
      proj = true;
      c_targ = c_star;
    } else if (mode == Mode::RESTORED) {
      if (step < cfg.release_step) {
        proj = true;
        c_targ = 0.0;
      } else {
        proj = true;
        c_targ = c_star;
      }
    }

    if (step % cfg.eval_every == 0) {
      EvalStats st = evaluate_model(cfg, P, test_set, eval_sym, proj, c_targ);
      double loss_bits = st.step_loss / std::log(2.0);
      double c_ratio = (std::abs(c_star) > 1e-9) ? (st.c_sym / c_star) : st.c_sym;
      std::cout << "step=" << std::setw(4) << step
                << " loss/ln2=" << std::fixed << std::setprecision(4) << loss_bits
                << " 1step=" << std::setprecision(1) << st.one_step_acc << "%"
                << " roll=" << std::setprecision(1) << st.rollout_acc << "%"
                << " c_sym=" << std::setprecision(3) << st.c_sym
                << " lag_n=" << std::setprecision(3) << st.lag_n_share
                << " Edist=" << std::setprecision(4) << st.E_dist
                << " Etrue=" << std::setprecision(4) << st.E_true << std::endl;
      if (ofs.is_open()) {
        ofs << step << "," << std::setprecision(6) << st.step_loss << "," << loss_bits << ","
            << st.one_step_acc << "," << st.rollout_acc << "," << st.c_sym << "," << c_ratio << ","
            << st.lag_n_share << "," << st.E_dist << "," << st.E_true << "\n";
        ofs.flush();
      }
    }

    if (step == cfg.total_steps) break;

    std::vector<Prompt> batch(cfg.batch_size), sym_batch;
    for (int m = 0; m < cfg.batch_size; ++m) batch[m] = sample_prompt(data_rng, cfg.n, cfg.sigma, p);
    sample_antithetic_probe(sym_rng, cfg.n, cfg.sigma, cfg.sym_quads, sym_batch);

    compute_batch_loss_and_grad(cfg, P, batch, sym_batch, proj, c_targ, grad, train_caches, sym_caches, bws);

    // AdamW parameter update
    double bc1 = 1.0 - std::pow(b1, step + 1);
    double bc2 = 1.0 - std::pow(b2, step + 1);
    for (const auto& b : P.blocks) {
      if (b.name.rfind("B1", 0) == 0 && b.is_lag_bias) continue;
      for (size_t i = 0; i < b.size; ++i) {
        if (b.is_lag_bias && i == 0) continue; // \beta_0 == 0 fixed
        size_t idx = b.offset + i;
        double g = grad[idx];
        m_buf[idx] = b1 * m_buf[idx] + (1.0 - b1) * g;
        v_buf[idx] = b2 * v_buf[idx] + (1.0 - b2) * g * g;
        double mh = m_buf[idx] / bc1;
        double vh = v_buf[idx] / bc2;
        if (!b.no_wd) P.data[idx] *= (1.0 - cfg.lr * cfg.weight_decay);
        P.data[idx] -= cfg.lr * mh / (std::sqrt(vh) + eps_adam);
      }
    }
  }
  return 0;
}
