// Comprehensive C++20 / Apple Accelerate full-model experiment engine for Priorities 2B, 3, 4, and 5:
//   - Modes:
//       0 = Natural (V(0) = 0, AdamW)
//       1 = Suppressed (\Pi_0^{(p)}, AdamW)
//       2 = External Offset (+c* x_{t-1}, V(0) = 0, AdamW)
//       3 = Restored (\Pi_0^{(p)} until release_step=600, then released)
//       4 = Weight-Calibrated (V(0) set along state axis to realize c*)
//       5 = ExtOffset + Nonzero V(0) (V(0) ~ N(0, 0.15^2) projected orthogonal to state axis so initial c_hat = c*)
//       6 = ExtOffset Transient (External offset for step < 300, then removed)
//       7 = ExtOffset + 10x lr_V (External offset with 10x learning rate on V)
//       8 = Natural with Stateless Block-RMS SGD (lr = 2e-3)
//       9 = Outcome-only supervision with Stateless Block-RMS SGD (lr = 2e-3)
//      10 = Outcome-only supervision with AdamW (lr = 1e-3, wd = 0.01)
//   - Diagnostics logged at checkpoints:
//       loss_bits, one_step_acc, rollout_acc, c_hat,
//       lag_n_share (within-letter), prefix_lag_n (full-prefix), prefix_lag_1 (full-prefix),
//       E_dist, E_true, norm_V, gate_proxy, qk_score_rms, b1_probe_acc (Block-1 match-bit linear probe!)

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include "cblas_shim.h"
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct Config {
  int n = 8;
  int sigma = 4;
  int d_model = 64;
  int n_heads = 3;
  int d_k = 24;
  int d_mlp = 128;
  float init_scale = 0.02f;
  float lr = 1e-3f;
  float weight_decay = 0.01f;  // AdamW decoupled decay, matches run_causal_fast / Table 18
  int batch_size = 256;
  int total_steps = 2400;
  int release_step = 600;
  int test_size = 1000;
  bool fast_probe = false;
};

struct Prompt {
  std::vector<int> tokens;
  std::vector<int> states;
  std::vector<int> bits;
  int w_star = 0;
};

struct Params {
  int vocab_size, seq_len, d_model, n_heads, d_k, d_mlp;
  std::vector<float> data;
  struct BlockSlice {
    std::string name;
    size_t offset, size;
    bool is_lag_bias, no_wd;
  };
  std::vector<BlockSlice> blocks;
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
    for (int l = 0; l < 2; ++l) {
      for (int h = 0; h < n_heads; ++h) {
        off_beta[l][h] = add("B" + std::to_string(l + 1) + "H" + std::to_string(h + 1) + "_beta",
                             (size_t)seq_len, true, true);
      }
    }
    data.assign(cur, 0.0f);
  }

  void init_random(uint64_t seed, float scale, bool zero_V = true) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, (double)scale);
    for (const auto& b : blocks) {
      if (b.is_lag_bias || (zero_V && b.name == "V")) {
        std::fill(data.begin() + b.offset, data.begin() + b.offset + b.size, 0.0f);
      } else {
        for (size_t i = 0; i < b.size; ++i) data[b.offset + i] = (float)nd(rng);
      }
    }
  }
};

inline void sgemm_nt(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, N);
}
inline void sgemm_nn(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
}
inline void sgemm_tn(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C, N);
}

struct ChunkBuffer {
  int max_M, L, D, DK, DM;
  std::vector<int> tokens;
  std::vector<float> h0;
  std::vector<float> q[2][3], k[2][3], v[2][3], alpha[2][3], raw_scores_b2[3];
  std::vector<float> htilde[2], u_mlp[2], r_mlp[2], h_out[2];
  std::vector<float> logits;
  std::vector<float> dh_out[2], dhtilde[2], du_mlp[2], dr_mlp[2];
  std::vector<float> dq, dk, dv;

  void init(const Config& cfg, int max_seqs) {
    max_M = max_seqs;
    L = 2 * cfg.n + 2;
    D = cfg.d_model;
    DK = cfg.d_k;
    DM = cfg.d_mlp;
    int ML = max_M * L;
    tokens.resize(ML);
    h0.resize(ML * D);
    for (int l = 0; l < 2; ++l) {
      for (int h = 0; h < 3; ++h) {
        q[l][h].resize(ML * DK);
        k[l][h].resize(ML * DK);
        v[l][h].resize(ML * D);
        alpha[l][h].assign(max_M * L * L, 0.0f);
      }
      htilde[l].resize(ML * D);
      u_mlp[l].resize(ML * DM);
      r_mlp[l].resize(ML * DM);
      h_out[l].resize(ML * D);
      dh_out[l].resize(ML * D);
      dhtilde[l].resize(ML * D);
      du_mlp[l].resize(ML * DM);
      dr_mlp[l].resize(ML * DM);
    }
    for (int h = 0; h < 3; ++h) raw_scores_b2[h].assign(max_M * L * L, 0.0f);
    logits.resize(ML * 2);
    dq.resize(ML * DK);
    dk.resize(ML * DK);
    dv.resize(ML * D);
  }
};

void forward_chunk(const Params& P, const Prompt* prompts, int M, ChunkBuffer& CB, bool save_raw_b2 = false) {
  if (M <= 0) return;
  const int L = P.seq_len, ML = M * L;
  const int D = P.d_model, DK = P.d_k, DM = P.d_mlp;
  const float inv_sqrt_dk = 1.0f / std::sqrt((float)DK);

  const float* E_tok = P.data.data() + P.off_E_tok;
  const float* E_pos = P.data.data() + P.off_E_pos;
  for (int m = 0; m < M; ++m) {
    const int* ptok = prompts[m].tokens.data();
    for (int i = 0; i < L; ++i) {
      int tok = ptok[i];
      CB.tokens[m * L + i] = tok;
      const float* et = E_tok + tok * D;
      const float* ep = E_pos + i * D;
      float* dst = CB.h0.data() + (m * L + i) * D;
      for (int d = 0; d < D; ++d) dst[d] = et[d] + ep[d];
    }
  }

  for (int l = 0; l < 2; ++l) {
    const float* hin = (l == 0) ? CB.h0.data() : CB.h_out[0].data();
    std::memcpy(CB.htilde[l].data(), hin, sizeof(float) * ML * D);

    for (int h = 0; h < 3; ++h) {
      const float* WQ = P.data.data() + P.off_WQ[l][h];
      const float* WK = P.data.data() + P.off_WK[l][h];
      const float* WV = P.data.data() + P.off_WV[l][h];
      const float* beta = P.data.data() + P.off_beta[l][h];

      sgemm_nt(ML, DK, D, 1.0f, hin, WQ, 0.0f, CB.q[l][h].data());
      sgemm_nt(ML, DK, D, 1.0f, hin, WK, 0.0f, CB.k[l][h].data());
      sgemm_nt(ML, D, D, 1.0f, hin, WV, 0.0f, CB.v[l][h].data());

      for (int m = 0; m < M; ++m) {
        const float* qm = CB.q[l][h].data() + m * L * DK;
        const float* km = CB.k[l][h].data() + m * L * DK;
        const float* vm = CB.v[l][h].data() + m * L * D;
        float* Am = CB.alpha[l][h].data() + m * L * L;
        float* htm = CB.htilde[l].data() + m * L * D;

        for (int i = 0; i < L; ++i) {
          const float* qi = qm + i * DK;
          float* row = Am + i * L;
          float max_s = -1e30f;
          for (int j = 0; j <= i; ++j) {
            const float* kj = km + j * DK;
            float dot = 0.0f;
            for (int d = 0; d < DK; ++d) dot += qi[d] * kj[d];
            float psi = dot * inv_sqrt_dk;
            if (save_raw_b2 && l == 1) CB.raw_scores_b2[h][m * L * L + i * L + j] = psi;
            int lag = i - j;
            if (l == 1 && lag > 0) psi += beta[lag];
            row[j] = psi;
            if (psi > max_s) max_s = psi;
          }
          float sum_exp = 0.0f;
          for (int j = 0; j <= i; ++j) {
            float e = std::exp(row[j] - max_s);
            row[j] = e;
            sum_exp += e;
          }
          float inv_sum = 1.0f / sum_exp;
          float* hti = htm + i * D;
          for (int j = 0; j <= i; ++j) {
            float w = row[j] * inv_sum;
            row[j] = w;
            const float* vj = vm + j * D;
            for (int d = 0; d < D; ++d) hti[d] += w * vj[d];
          }
          for (int j = i + 1; j < L; ++j) row[j] = 0.0f;
        }
      }
    }

    const float* Wup = P.data.data() + P.off_Wup[l];
    const float* bup = P.data.data() + P.off_bup[l];
    const float* Wdown = P.data.data() + P.off_Wdown[l];
    sgemm_nt(ML, DM, D, 1.0f, CB.htilde[l].data(), Wup, 0.0f, CB.u_mlp[l].data());
    for (int row = 0; row < ML; ++row) {
      float* ui = CB.u_mlp[l].data() + row * DM;
      float* ri = CB.r_mlp[l].data() + row * DM;
      for (int c = 0; c < DM; ++c) {
        float val = ui[c] + bup[c];
        ui[c] = val;
        ri[c] = (val > 0.0f) ? val : 0.0f;
      }
    }
    std::memcpy(CB.h_out[l].data(), CB.htilde[l].data(), sizeof(float) * ML * D);
    sgemm_nt(ML, D, DM, 1.0f, CB.r_mlp[l].data(), Wdown, 1.0f, CB.h_out[l].data());
  }

  const float* V = P.data.data() + P.off_V;
  sgemm_nt(ML, 2, D, 1.0f, CB.h_out[1].data(), V, 0.0f, CB.logits.data());
}

void backward_chunk(const Params& P, int M, ChunkBuffer& CB, const float* d_logits, float* grad) {
  if (M <= 0) return;
  const int L = P.seq_len, ML = M * L;
  const int D = P.d_model, DK = P.d_k, DM = P.d_mlp;
  const float inv_sqrt_dk = 1.0f / std::sqrt((float)DK);

  const float* V = P.data.data() + P.off_V;
  sgemm_tn(2, D, ML, 1.0f, d_logits, CB.h_out[1].data(), 1.0f, grad + P.off_V);
  sgemm_nn(ML, D, 2, 1.0f, d_logits, V, 0.0f, CB.dh_out[1].data());

  float dA[64 * 64]; // supports up to n=30 (L=62)
  for (int l = 1; l >= 0; --l) {
    const float* hin = (l == 0) ? CB.h0.data() : CB.h_out[0].data();
    const float* Wup = P.data.data() + P.off_Wup[l];
    const float* Wdown = P.data.data() + P.off_Wdown[l];

    std::memcpy(CB.dhtilde[l].data(), CB.dh_out[l].data(), sizeof(float) * ML * D);
    sgemm_tn(D, DM, ML, 1.0f, CB.dh_out[l].data(), CB.r_mlp[l].data(), 1.0f, grad + P.off_Wdown[l]);
    sgemm_nn(ML, DM, D, 1.0f, CB.dh_out[l].data(), Wdown, 0.0f, CB.dr_mlp[l].data());

    float* dbup = grad + P.off_bup[l];
    for (int row = 0; row < ML; ++row) {
      const float* ui = CB.u_mlp[l].data() + row * DM;
      const float* dri = CB.dr_mlp[l].data() + row * DM;
      float* dui = CB.du_mlp[l].data() + row * DM;
      for (int c = 0; c < DM; ++c) {
        float g = (ui[c] > 0.0f) ? dri[c] : 0.0f;
        dui[c] = g;
        dbup[c] += g;
      }
    }
    sgemm_tn(DM, D, ML, 1.0f, CB.du_mlp[l].data(), CB.htilde[l].data(), 1.0f, grad + P.off_Wup[l]);
    sgemm_nn(ML, D, DM, 1.0f, CB.du_mlp[l].data(), Wup, 1.0f, CB.dhtilde[l].data());

    float* dhin = (l == 1) ? CB.dh_out[0].data() : CB.dh_out[1].data();
    std::memcpy(dhin, CB.dhtilde[l].data(), sizeof(float) * ML * D);

    for (int h = 0; h < 3; ++h) {
      const float* WQ = P.data.data() + P.off_WQ[l][h];
      const float* WK = P.data.data() + P.off_WK[l][h];
      const float* WV = P.data.data() + P.off_WV[l][h];
      float* dbeta = grad + P.off_beta[l][h];

      std::fill(CB.dq.begin(), CB.dq.begin() + ML * DK, 0.0f);
      std::fill(CB.dk.begin(), CB.dk.begin() + ML * DK, 0.0f);
      std::fill(CB.dv.begin(), CB.dv.begin() + ML * D, 0.0f);

      for (int m = 0; m < M; ++m) {
        const float* Am = CB.alpha[l][h].data() + m * L * L;
        const float* dhtm = CB.dhtilde[l].data() + m * L * D;
        const float* qm = CB.q[l][h].data() + m * L * DK;
        const float* km = CB.k[l][h].data() + m * L * DK;
        const float* vm = CB.v[l][h].data() + m * L * D;
        float* dqm = CB.dq.data() + m * L * DK;
        float* dkm = CB.dk.data() + m * L * DK;
        float* dvm = CB.dv.data() + m * L * D;

        for (int i = 0; i < L; ++i) {
          const float* dht_i = dhtm + i * D;
          float dot_adA = 0.0f;
          for (int j = 0; j <= i; ++j) {
            float w = Am[i * L + j];
            float* dv_j = dvm + j * D;
            const float* v_j = vm + j * D;
            float da_ij = 0.0f;
            for (int d = 0; d < D; ++d) {
              dv_j[d] += w * dht_i[d];
              da_ij += dht_i[d] * v_j[d];
            }
            dA[i * L + j] = da_ij;
            dot_adA += w * da_ij;
          }
          float* dq_i = dqm + i * DK;
          const float* q_i = qm + i * DK;
          for (int j = 0; j <= i; ++j) {
            float ds = Am[i * L + j] * (dA[i * L + j] - dot_adA);
            int lag = i - j;
            if (l == 1 && lag > 0) dbeta[lag] += ds;
            float ds_sc = ds * inv_sqrt_dk;
            const float* k_j = km + j * DK;
            float* dk_j = dkm + j * DK;
            for (int d = 0; d < DK; ++d) {
              dq_i[d] += ds_sc * k_j[d];
              dk_j[d] += ds_sc * q_i[d];
            }
          }
        }
      }

      sgemm_tn(DK, D, ML, 1.0f, CB.dq.data(), hin, 1.0f, grad + P.off_WQ[l][h]);
      sgemm_tn(DK, D, ML, 1.0f, CB.dk.data(), hin, 1.0f, grad + P.off_WK[l][h]);
      sgemm_tn(D, D, ML, 1.0f, CB.dv.data(), hin, 1.0f, grad + P.off_WV[l][h]);
      sgemm_nn(ML, D, DK, 1.0f, CB.dq.data(), WQ, 1.0f, dhin);
      sgemm_nn(ML, D, DK, 1.0f, CB.dk.data(), WK, 1.0f, dhin);
      sgemm_nn(ML, D, D, 1.0f, CB.dv.data(), WV, 1.0f, dhin);
    }
  }

  const float* dh0 = CB.dh_out[1].data();
  float* dE_tok = grad + P.off_E_tok;
  float* dE_pos = grad + P.off_E_pos;
  for (int m = 0; m < M; ++m) {
    for (int i = 0; i < L; ++i) {
      int row = m * L + i;
      const float* g = dh0 + row * D;
      float* dt = dE_tok + CB.tokens[row] * D;
      float* dp = dE_pos + i * D;
      for (int d = 0; d < D; ++d) {
        dt[d] += g[d];
        dp[d] += g[d];
      }
    }
  }
}

void forward_chunk_single_pos(const Params& P, int pos, const int* step_toks, int M, ChunkBuffer& CB,
                              std::vector<float>& pos_hin, std::vector<float>& pos_q,
                              std::vector<float>& pos_k, std::vector<float>& pos_v,
                              std::vector<float>& pos_u, std::vector<float>& pos_r,
                              std::vector<float>& pos_ho, std::vector<float>& pos_z) {
  const int L = P.seq_len, D = P.d_model, DK = P.d_k, DM = P.d_mlp;
  const float inv_sqrt_dk = 1.0f / std::sqrt((float)DK);

  const float* E_tok = P.data.data() + P.off_E_tok;
  const float* ep = P.data.data() + P.off_E_pos + pos * D;
  for (int m = 0; m < M; ++m) {
    int tok = step_toks[m];
    CB.tokens[m * L + pos] = tok;
    const float* et = E_tok + tok * D;
    float* dst = CB.h0.data() + (m * L + pos) * D;
    for (int d = 0; d < D; ++d) {
      float val = et[d] + ep[d];
      dst[d] = val;
      pos_hin[m * D + d] = val;
    }
  }

  for (int l = 0; l < 2; ++l) {
    if (l == 1) {
      for (int m = 0; m < M; ++m) {
        std::memcpy(pos_hin.data() + m * D, CB.h_out[0].data() + (m * L + pos) * D, sizeof(float) * D);
      }
    }
    std::memcpy(pos_ho.data(), pos_hin.data(), sizeof(float) * M * D);

    for (int h = 0; h < 3; ++h) {
      const float* WQ = P.data.data() + P.off_WQ[l][h];
      const float* WK = P.data.data() + P.off_WK[l][h];
      const float* WV = P.data.data() + P.off_WV[l][h];
      const float* beta = P.data.data() + P.off_beta[l][h];
      sgemm_nt(M, DK, D, 1.0f, pos_hin.data(), WQ, 0.0f, pos_q.data());
      sgemm_nt(M, DK, D, 1.0f, pos_hin.data(), WK, 0.0f, pos_k.data());
      sgemm_nt(M, D, D, 1.0f, pos_hin.data(), WV, 0.0f, pos_v.data());

      for (int m = 0; m < M; ++m) {
        std::memcpy(CB.k[l][h].data() + (m * L + pos) * DK, pos_k.data() + m * DK, sizeof(float) * DK);
        std::memcpy(CB.v[l][h].data() + (m * L + pos) * D, pos_v.data() + m * D, sizeof(float) * D);
        const float* qi = pos_q.data() + m * DK;
        const float* km = CB.k[l][h].data() + m * L * DK;
        const float* vm = CB.v[l][h].data() + m * L * D;
        float scores[64];
        float max_s = -1e30f;
        for (int j = 0; j <= pos; ++j) {
          const float* kj = km + j * DK;
          float dot = 0.0f;
          for (int d = 0; d < DK; ++d) dot += qi[d] * kj[d];
          int lag = pos - j;
          float b_lag = (l == 1 && lag > 0) ? beta[lag] : 0.0f;
          float s = b_lag + dot * inv_sqrt_dk;
          scores[j] = s;
          if (s > max_s) max_s = s;
        }
        float sum_exp = 0.0f;
        for (int j = 0; j <= pos; ++j) {
          scores[j] = std::exp(scores[j] - max_s);
          sum_exp += scores[j];
        }
        float inv_sum = 1.0f / sum_exp;
        float* hti = pos_ho.data() + m * D;
        for (int j = 0; j <= pos; ++j) {
          float w = scores[j] * inv_sum;
          const float* vj = vm + j * D;
          for (int d = 0; d < D; ++d) hti[d] += w * vj[d];
        }
      }
    }

    const float* Wup = P.data.data() + P.off_Wup[l];
    const float* bup = P.data.data() + P.off_bup[l];
    const float* Wdown = P.data.data() + P.off_Wdown[l];
    sgemm_nt(M, DM, D, 1.0f, pos_ho.data(), Wup, 0.0f, pos_u.data());
    for (int m = 0; m < M; ++m) {
      float* ui = pos_u.data() + m * DM;
      float* ri = pos_r.data() + m * DM;
      for (int c = 0; c < DM; ++c) {
        float val = ui[c] + bup[c];
        ri[c] = (val > 0.0f) ? val : 0.0f;
      }
    }
    sgemm_nt(M, D, DM, 1.0f, pos_r.data(), Wdown, 1.0f, pos_ho.data());
    for (int m = 0; m < M; ++m) {
      std::memcpy(CB.h_out[l].data() + (m * L + pos) * D, pos_ho.data() + m * D, sizeof(float) * D);
    }
  }
  const float* V = P.data.data() + P.off_V;
  sgemm_nt(M, 2, D, 1.0f, pos_ho.data(), V, 0.0f, pos_z.data());
  for (int m = 0; m < M; ++m) {
    CB.logits[(m * L + pos) * 2 + 0] = pos_z[m * 2 + 0];
    CB.logits[(m * L + pos) * 2 + 1] = pos_z[m * 2 + 1];
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
    int wt = match ? pr.w_star : [&]() {
      int r = nonmatch_dist(rng);
      return (r >= pr.w_star) ? (r + 1) : r;
    }();
    int bt = match ? 1 : 0;
    pr.bits[t - 1] = bt;
    pr.states[t] = (pr.states[t - 1] + bt) & 1;
    pr.tokens[1 + t] = 2 + wt;
  }
  pr.tokens[n + 2] = 2 + sigma;
  for (int t = 2; t <= n; ++t) pr.tokens[n + 1 + t] = pr.states[t - 1];
  return pr;
}

// Linear ridge probe on Block-1 output h^{(1)}_{1+t} predicting b_t = 1[w_t == w*]
// Fits w in R^{D+1} on probe_train (first 400 prompts of test_set) and evaluates on remaining 600 prompts!
double compute_block1_match_probe_acc(const Config& cfg, const Params& P, const std::vector<Prompt>& test_set, ChunkBuffer& CB) {
  const int N = (int)test_set.size();
  const int n = cfg.n;
  const int L = P.seq_len;
  const int D = P.d_model;
  const int Pdim = D + 1; // +1 bias
  int N_tr = N / 2;
  int N_te = N - N_tr;

  std::vector<double> XtX(Pdim * Pdim, 0.0);
  std::vector<double> Xty(Pdim, 0.0);
  for (int i = 0; i < Pdim; ++i) XtX[i * Pdim + i] = 1e-2;

  for (int m = 0; m < N_tr; ++m) {
    for (int t = 1; t <= n; ++t) {
      const float* h = CB.h_out[0].data() + (m * L + (1 + t)) * D;
      double y = test_set[m].bits[t - 1] ? 1.0 : -1.0;
      for (int i = 0; i < D; ++i) {
        double hi = h[i];
        Xty[i] += hi * y;
        for (int j = i; j < D; ++j) XtX[i * Pdim + j] += hi * h[j];
        XtX[i * Pdim + D] += hi;
      }
      Xty[D] += y;
      XtX[D * Pdim + D] += 1.0;
    }
  }
  for (int i = 0; i < Pdim; ++i) {
    for (int j = 0; j < i; ++j) XtX[i * Pdim + j] = XtX[j * Pdim + i];
  }

  // Gaussian elimination with partial pivoting on Pdim x Pdim (65 x 65)
  std::vector<double> w = Xty;
  for (int k = 0; k < Pdim; ++k) {
    int piv = k;
    for (int i = k + 1; i < Pdim; ++i) {
      if (std::abs(XtX[i * Pdim + k]) > std::abs(XtX[piv * Pdim + k])) piv = i;
    }
    if (piv != k) {
      for (int j = k; j < Pdim; ++j) std::swap(XtX[k * Pdim + j], XtX[piv * Pdim + j]);
      std::swap(w[k], w[piv]);
    }
    double diag = XtX[k * Pdim + k];
    if (std::abs(diag) < 1e-12) continue;
    for (int i = k + 1; i < Pdim; ++i) {
      double f = XtX[i * Pdim + k] / diag;
      w[i] -= f * w[k];
      for (int j = k; j < Pdim; ++j) XtX[i * Pdim + j] -= f * XtX[k * Pdim + j];
    }
  }
  for (int k = Pdim - 1; k >= 0; --k) {
    double s = w[k];
    for (int j = k + 1; j < Pdim; ++j) s -= XtX[k * Pdim + j] * w[j];
    w[k] = (std::abs(XtX[k * Pdim + k]) > 1e-12) ? (s / XtX[k * Pdim + k]) : 0.0;
  }

  long long correct = 0;
  for (int m = N_tr; m < N; ++m) {
    for (int t = 1; t <= n; ++t) {
      const float* h = CB.h_out[0].data() + (m * L + (1 + t)) * D;
      double pred = w[D];
      for (int i = 0; i < D; ++i) pred += w[i] * h[i];
      int yhat = (pred >= 0.0) ? 1 : 0;
      if (yhat == test_set[m].bits[t - 1]) correct++;
    }
  }
  return 100.0 * (double)correct / ((double)N_te * n);
}

struct EvalStats {
  double step_loss = 0.0;
  double one_step_acc = 0.0;
  double rollout_acc = 0.0;
  double c_hat = 0.0;
  double lag_n_share = 0.0;   // max head within-letter share on lag n
  double prefix_lag_n = 0.0;  // mean across heads & steps of full-prefix weight on lag n
  double prefix_lag_1 = 0.0;  // mean across heads & steps of full-prefix weight on lag 1
  double E_dist = 0.0;
  double E_true = 0.0;
  double norm_V = 0.0;
  double gate_proxy = 0.0;
  double qk_score_rms = 0.0;
  double b1_probe_acc = 0.0;
};

EvalStats evaluate_model(
    const Config& cfg,
    const Params& P,
    const std::vector<Prompt>& test_set,
    bool suppress_c0,
    float add_c_offset,
    ChunkBuffer& CB,
    bool outcome_only = false,
    bool compute_probe = true) {

  const int N = (int)test_set.size();
  const int n = cfg.n;
  const int L = P.seq_len;

  forward_chunk(P, test_set.data(), N, CB, true);

  EvalStats st;
  // ||V||_F
  double nv2 = 0.0;
  for (int i = 0; i < 2 * cfg.d_model; ++i) {
    double v = P.data[P.off_V + i];
    nv2 += v * v;
  }
  st.norm_V = std::sqrt(nv2);
  st.b1_probe_acc = compute_probe ? compute_block1_match_probe_acc(cfg, P, test_set, CB) : 0.0;

  if (outcome_only) {
    const int pos = n + 2;
    long long correct = 0, agree_s0 = 0;
    double p_true_sum = 0.0, c0_sum = 0.0, qk_sq = 0.0;
    for (int m = 0; m < N; ++m) {
      const Prompt& pr = test_set[m];
      double z0 = CB.logits[(m * L + pos) * 2 + 0];
      double z1 = CB.logits[(m * L + pos) * 2 + 1];
      double mx = std::max(z0, z1);
      double e0 = std::exp(z0 - mx), e1 = std::exp(z1 - mx);
      double p0 = e0 / (e0 + e1), p1 = e1 / (e0 + e1);
      int y = pr.states[n];
      double p_true = (y == 0) ? p0 : p1;
      st.step_loss += -std::log(p_true + 1e-300) / (double)N;
      p_true_sum += p_true / (double)N;
      int pred = (z0 >= z1) ? 0 : 1;
      if (pred == y) correct++;
      if (pred == pr.states[0]) agree_s0++;
      double x0 = (pr.states[0] == 0) ? 1.0 : -1.0;
      c0_sum += x0 * (z0 - z1) / (double)N;
      for (int h = 0; h < 3; ++h) {
        const float* Psi = CB.raw_scores_b2[h].data() + (m * L + pos) * L;
        for (int r = 1; r <= n; ++r) {
          qk_sq += ((double)Psi[1 + r] * Psi[1 + r]) / (3.0 * N * n);
        }
      }
    }
    st.one_step_acc = 100.0 * (double)correct / (double)N;
    st.rollout_acc = st.one_step_acc;
    st.c_hat = c0_sum;
    st.E_true = p_true_sum;                          // E[p_theta(s_n)] under outcome supervision
    st.E_dist = (double)agree_s0 / (double)N;        // fraction agreeing with memoryless s_0 predictor
    st.qk_score_rms = std::sqrt(qk_sq);
    return st;
  }

  std::vector<double> c_hat_t(n + 1, 0.0);
  for (int m = 0; m < N; ++m) {
    for (int t = 1; t <= n; ++t) {
      int pos = n + 1 + t;
      double x_prev = (test_set[m].states[t - 1] == 0) ? 1.0 : -1.0;
      double dz = (double)CB.logits[(m * L + pos) * 2 + 0] -
                  (double)CB.logits[(m * L + pos) * 2 + 1] + 1.0 * add_c_offset * x_prev;
      c_hat_t[t] += x_prev * dz / N;
    }
  }
  for (int t = 2; t <= n; ++t) st.c_hat += (suppress_c0 ? 0.0 : c_hat_t[t]) / (n - 1);

  long long tot_1step = 0, tot_roll = 0;
  double head_lag[3] = {0.0, 0.0, 0.0};
  double pfx_lag_n = 0.0, pfx_lag_1 = 0.0;
  double gate_sq = 0.0, qk_sq = 0.0;
  std::vector<double> corr((n + 1) * (n + 1), 0.0);

  for (int m = 0; m < N; ++m) {
    const Prompt& pr = test_set[m];
    for (int t = 1; t <= n; ++t) {
      int pos = n + 1 + t;
      double x_prev = (pr.states[t - 1] == 0) ? 1.0 : -1.0;
      double z0 = CB.logits[(m * L + pos) * 2 + 0] + 0.5 * add_c_offset * x_prev;
      double z1 = CB.logits[(m * L + pos) * 2 + 1] - 0.5 * add_c_offset * x_prev;
      if (suppress_c0) {
        double shift = -0.5 * c_hat_t[t] * x_prev;
        z0 += shift;
        z1 -= shift;
      }
      double mx = std::max(z0, z1);
      double e0 = std::exp(z0 - mx), e1 = std::exp(z1 - mx);
      double p0 = e0 / (e0 + e1), p1 = e1 / (e0 + e1);
      int y = pr.states[t];
      double p_true = (y == 0) ? p0 : p1;
      st.step_loss += -std::log(p_true + 1e-300) / ((double)N * n);
      if (((z0 >= z1) ? 0 : 1) == y) tot_1step++;

      if (t >= 2) {
        double dz_eff = z0 - z1;
        double c_eff = suppress_c0 ? 0.0 : c_hat_t[t];
        double non_state = dz_eff - c_eff * x_prev;
        gate_sq += (non_state * non_state) / ((double)N * (n - 1));

        double res_mag = 1.0 - p_true;
        double chi_t = pr.bits[t - 1] ? -1.0 : 1.0;
        for (int r = 1; r <= n; ++r) {
          double chi_r = pr.bits[r - 1] ? -1.0 : 1.0;
          corr[t * (n + 1) + r] += (res_mag * chi_t * chi_r) / N;
        }
        for (int h = 0; h < 3; ++h) {
          const float* A = CB.alpha[1][h].data() + (m * L + pos) * L;
          const float* Psi = CB.raw_scores_b2[h].data() + (m * L + pos) * L;
          double let_sum = 0.0;
          for (int r = 1; r <= n; ++r) {
            let_sum += A[1 + r];
            qk_sq += ((double)Psi[1 + r] * Psi[1 + r]) / (3.0 * N * (n - 1) * n);
          }
          double pi_star = (let_sum > 1e-12) ? (A[1 + t] / let_sum) : (1.0 / n);
          head_lag[h] += pi_star / ((double)N * (n - 1));
          pfx_lag_n += A[pos - n] / (3.0 * N * (n - 1)); // lag n = 1+t
          pfx_lag_1 += A[pos - 1] / (3.0 * N * (n - 1)); // lag 1 = pos-1
        }
      }
    }
  }

  // Greedy rollout across N prompts
  std::vector<int> prev_s(N);
  std::vector<bool> ok(N, true);
  for (int m = 0; m < N; ++m) prev_s[m] = test_set[m].states[0];
  std::vector<float> p_hin(N * P.d_model), p_q(N * P.d_k), p_k(N * P.d_k), p_v(N * P.d_model);
  std::vector<float> p_u(N * P.d_mlp), p_r(N * P.d_mlp), p_ho(N * P.d_model), p_z(N * 2);

  for (int t = 1; t <= n; ++t) {
    int pos = n + 1 + t;
    if (t >= 2) {
      forward_chunk_single_pos(P, pos, prev_s.data(), N, CB, p_hin, p_q, p_k, p_v, p_u, p_r, p_ho, p_z);
    }
    for (int m = 0; m < N; ++m) {
      double x_prev = (prev_s[m] == 0) ? 1.0 : -1.0;
      double z0 = CB.logits[(m * L + pos) * 2 + 0] + 0.5 * add_c_offset * x_prev;
      double z1 = CB.logits[(m * L + pos) * 2 + 1] - 0.5 * add_c_offset * x_prev;
      if (suppress_c0) {
        double shift = -0.5 * c_hat_t[t] * x_prev;
        z0 += shift;
        z1 -= shift;
      }
      int pred = (z0 >= z1) ? 0 : 1;
      if (pred != test_set[m].states[t]) ok[m] = false;
      prev_s[m] = pred;
    }
  }
  for (int m = 0; m < N; ++m) if (ok[m]) tot_roll++;

  st.one_step_acc = 100.0 * (double)tot_1step / ((double)N * n);
  st.rollout_acc = 100.0 * (double)tot_roll / (double)N;
  st.lag_n_share = std::max({head_lag[0], head_lag[1], head_lag[2]});
  st.prefix_lag_n = pfx_lag_n;
  st.prefix_lag_1 = pfx_lag_1;
  st.gate_proxy = std::sqrt(gate_sq);
  st.qk_score_rms = std::sqrt(qk_sq);

  double ed_sum = 0.0, et_sum = 0.0;
  for (int t = 2; t <= n; ++t) {
    et_sum += corr[t * (n + 1) + t];
    for (int r = 1; r <= n; ++r) {
      if (r != t) ed_sum += std::abs(corr[t * (n + 1) + r]);
    }
  }
  st.E_true = et_sum / (n - 1);
  st.E_dist = ed_sum / ((double)(n - 1) * (n - 1));
  return st;
}

bool should_eval_step(int step, int total_steps) {
  if (step == total_steps) return true;
  const int ckpts[] = {0, 25, 50, 100, 200, 300, 400, 600, 800, 1000, 1200, 1600, 2000, 2400, 3200, 4000, 6000, 8000};
  for (int c : ckpts) if (step == c) return true;
  return false;
}

void run_single(const Config& cfg, double p, int seed, int mode, const std::string& csv_path) {
  const float c_star = (float)std::log((1.0 - p) / p);
  Params P;
  P.allocate(cfg);
  P.init_random(seed, cfg.init_scale, true);

  const int D = cfg.d_model;
  const float* e0 = P.data.data() + P.off_E_tok + 0 * D;
  const float* e1 = P.data.data() + P.off_E_tok + 1 * D;
  double vs_norm2 = 0.0;
  std::vector<double> vs(D);
  for (int d = 0; d < D; ++d) {
    vs[d] = 0.5 * ((double)e0[d] - (double)e1[d]);
    vs_norm2 += vs[d] * vs[d];
  }

  if (mode == 4 && std::abs(c_star) > 1e-7f && vs_norm2 > 1e-12) {
    // Weight-Calibrated V(0) = c*
    for (int d = 0; d < D; ++d) {
      float w = (float)(0.5 * (double)c_star * vs[d] / vs_norm2);
      P.data[P.off_V + 0 * D + d] = +w;
      P.data[P.off_V + 1 * D + d] = -w;
    }
  } else if (mode == 5) {
    // Variant B1: External Offset + Nonzero V(0) orthogonal to state axis (||V(0)|| ~ 0.15 * sqrt(2D) ~ 1.7, matching Natural's ||V||!)
    std::mt19937_64 vrng(999999 + seed);
    std::normal_distribution<double> vnd(0.0, 0.15);
    std::vector<double> vrow(D);
    double dot = 0.0;
    for (int d = 0; d < D; ++d) {
      vrow[d] = vnd(vrng);
      dot += vrow[d] * vs[d];
    }
    for (int d = 0; d < D; ++d) {
      if (vs_norm2 > 1e-12) vrow[d] -= (dot / vs_norm2) * vs[d];
      P.data[P.off_V + 0 * D + d] = (float)(+vrow[d]);
      P.data[P.off_V + 1 * D + d] = (float)(-vrow[d]);
    }
  }

  std::mt19937_64 test_rng(100000 + seed);
  std::vector<Prompt> test_set(cfg.test_size);
  for (int m = 0; m < cfg.test_size; ++m) test_set[m] = sample_prompt(test_rng, cfg.n, cfg.sigma, p);

  std::mt19937_64 data_rng(200000 + seed);
  std::vector<float> m_buf(P.data.size(), 0.0f), v_buf(P.data.size(), 0.0f), grad(P.data.size(), 0.0f);
  const float b1 = 0.9f, b2 = 0.999f, eps_adam = 1e-8f;

  const int n = cfg.n;
  const int L = P.seq_len;
  const int B = cfg.batch_size;

  ChunkBuffer train_cb, eval_cb;
  train_cb.init(cfg, B);
  eval_cb.init(cfg, cfg.test_size);

  std::ofstream ofs(csv_path);
  ofs << "step,loss_bits,one_step_acc,rollout_acc,c_hat,lag_n_share,prefix_lag_n,prefix_lag_1,E_dist,E_true,norm_V,gate_proxy,qk_score_rms,b1_probe_acc\n";

  const bool outcome_only = (mode == 9 || mode == 10);
  if (outcome_only) {
    for (auto& pr : test_set) {
      for (int pos = n + 3; pos < L; ++pos) pr.tokens[pos] = 2 + cfg.sigma;
    }
  }

  for (int step = 0; step <= cfg.total_steps; ++step) {
    bool suppress_c0 = false;
    float add_c_offset = 0.0f;
    if (mode == 1) suppress_c0 = true;
    else if (mode == 2 || mode == 5 || mode == 7) add_c_offset = c_star;
    else if (mode == 6) add_c_offset = (step < 300) ? c_star : 0.0f;
    else if (mode == 3) suppress_c0 = (step < cfg.release_step);

    if (should_eval_step(step, cfg.total_steps)) {
      bool do_probe = !cfg.fast_probe || (step == 0 || step == cfg.total_steps);
      EvalStats st = evaluate_model(cfg, P, test_set, suppress_c0, add_c_offset, eval_cb, outcome_only, do_probe);
      ofs << step << "," << std::fixed << std::setprecision(5) << (st.step_loss / std::log(2.0)) << ","
          << std::setprecision(2) << st.one_step_acc << "," << st.rollout_acc << ","
          << std::setprecision(4) << st.c_hat << "," << st.lag_n_share << ","
          << st.prefix_lag_n << "," << st.prefix_lag_1 << ","
          << st.E_dist << "," << st.E_true << ","
          << st.norm_V << "," << st.gate_proxy << "," << st.qk_score_rms << ","
          << std::setprecision(2) << st.b1_probe_acc << "\n";
      ofs.flush();
    }
    if (step == cfg.total_steps) break;

    std::vector<Prompt> batch(B);
    for (int m = 0; m < B; ++m) {
      batch[m] = sample_prompt(data_rng, n, cfg.sigma, p);
      if (outcome_only) {
        for (int pos = n + 3; pos < L; ++pos) batch[m].tokens[pos] = 2 + cfg.sigma;
      }
    }
    forward_chunk(P, batch.data(), B, train_cb, false);

    std::vector<float> train_dlogits(B * L * 2, 0.0f);
    if (outcome_only) {
      const int pos = n + 2;
      const double inv_B = 1.0 / (double)B;
      for (int m = 0; m < B; ++m) {
        double z0 = (double)train_cb.logits[(m * L + pos) * 2 + 0];
        double z1 = (double)train_cb.logits[(m * L + pos) * 2 + 1];
        double mx = std::max(z0, z1);
        double e0 = std::exp(z0 - mx), e1 = std::exp(z1 - mx);
        double p0 = e0 / (e0 + e1), p1 = e1 / (e0 + e1);
        int y = batch[m].states[n];
        train_dlogits[(m * L + pos) * 2 + 0] = (float)((p0 - (y == 0 ? 1.0 : 0.0)) * inv_B);
        train_dlogits[(m * L + pos) * 2 + 1] = (float)((p1 - (y == 1 ? 1.0 : 0.0)) * inv_B);
      }
    } else {
      std::vector<double> c_hat_batch(n + 1, 0.0);
      if (suppress_c0) {
        for (int m = 0; m < B; ++m) {
          for (int t = 1; t <= n; ++t) {
            int pos = n + 1 + t;
            double x_prev = (batch[m].states[t - 1] == 0) ? 1.0 : -1.0;
            double dz = (double)train_cb.logits[(m * L + pos) * 2 + 0] -
                        (double)train_cb.logits[(m * L + pos) * 2 + 1];
            c_hat_batch[t] += x_prev * dz / B;
          }
        }
      }

      std::vector<double> mean_x_d(n + 1, 0.0);
      const double inv_Bn = 1.0 / ((double)B * n);

      for (int m = 0; m < B; ++m) {
        for (int t = 1; t <= n; ++t) {
          int pos = n + 1 + t;
          double x_prev = (batch[m].states[t - 1] == 0) ? 1.0 : -1.0;
          double z0 = (double)train_cb.logits[(m * L + pos) * 2 + 0] + 0.5 * add_c_offset * x_prev;
          double z1 = (double)train_cb.logits[(m * L + pos) * 2 + 1] - 0.5 * add_c_offset * x_prev;
          if (suppress_c0) {
            double shift = -0.5 * c_hat_batch[t] * x_prev;
            z0 += shift;
            z1 -= shift;
          }
          double mx = std::max(z0, z1);
          double e0 = std::exp(z0 - mx), e1 = std::exp(z1 - mx);
          double p0 = e0 / (e0 + e1), p1 = e1 / (e0 + e1);
          int y = batch[m].states[t];
          double g0 = (p0 - (y == 0 ? 1.0 : 0.0)) * inv_Bn;
          double g1 = (p1 - (y == 1 ? 1.0 : 0.0)) * inv_Bn;
          train_dlogits[(m * L + pos) * 2 + 0] = (float)g0;
          train_dlogits[(m * L + pos) * 2 + 1] = (float)g1;
          if (suppress_c0) mean_x_d[t] += (x_prev * (g0 - g1)) / B;
        }
      }

      if (suppress_c0) {
        for (int m = 0; m < B; ++m) {
          for (int t = 1; t <= n; ++t) {
            int pos = n + 1 + t;
            double x_prev = (batch[m].states[t - 1] == 0) ? 1.0 : -1.0;
            float proj_sub = (float)(0.5 * (mean_x_d[t] - (0.1 / ((double)B * n)) * c_hat_batch[t]) * x_prev);
            train_dlogits[(m * L + pos) * 2 + 0] -= proj_sub;
            train_dlogits[(m * L + pos) * 2 + 1] += proj_sub;
          }
        }
      }
    }

    std::fill(grad.begin(), grad.end(), 0.0f);
    backward_chunk(P, B, train_cb, train_dlogits.data(), grad.data());

    if (mode == 8 || mode == 9) {
      // Stateless Block-RMS SGD (lr_rms = 2e-3)
      const float lr_rms = 2e-3f;
      const size_t L_active = outcome_only ? (size_t)(n + 3) : (size_t)L;
      for (const auto& b : P.blocks) {
        if (b.name.rfind("B1", 0) == 0 && b.is_lag_bias) continue;
        size_t active_sz = b.size;
        if (outcome_only) {
          if (b.name == "E_pos") active_sz = L_active * (size_t)D;
          else if (b.is_lag_bias) active_sz = L_active;
        }
        double sq = 0.0;
        size_t cnt = 0;
        for (size_t i = 0; i < active_sz; ++i) {
          if (b.is_lag_bias && i == 0) continue;
          double g = grad[b.offset + i];
          sq += g * g;
          cnt++;
        }
        float rms = (float)(std::sqrt(sq / std::max<size_t>(1, cnt)) + 1e-8);
        for (size_t i = 0; i < active_sz; ++i) {
          if (b.is_lag_bias && i == 0) continue;
          P.data[b.offset + i] -= (lr_rms / rms) * grad[b.offset + i];
        }
      }
    } else {
      float bc1 = 1.0f - std::pow(b1, step + 1);
      float bc2 = 1.0f - std::pow(b2, step + 1);
      for (const auto& b : P.blocks) {
        if (b.name.rfind("B1", 0) == 0 && b.is_lag_bias) continue;
        float blk_lr = (mode == 7 && b.name == "V") ? (10.0f * cfg.lr) : cfg.lr;
        for (size_t i = 0; i < b.size; ++i) {
          if (b.is_lag_bias && i == 0) continue;
          size_t idx = b.offset + i;
          float g = grad[idx];
          m_buf[idx] = b1 * m_buf[idx] + (1.0f - b1) * g;
          v_buf[idx] = b2 * v_buf[idx] + (1.0f - b2) * g * g;
          float mh = m_buf[idx] / bc1;
          float vh = v_buf[idx] / bc2;
          if (!b.no_wd && cfg.weight_decay > 0.0f) P.data[idx] *= (1.0f - blk_lr * cfg.weight_decay);
          P.data[idx] -= blk_lr * mh / (std::sqrt(vh) + eps_adam);
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
  Config cfg;
  double p = 0.25;
  int seed = 42;
  int mode = 0;
  std::string out_dir = "results_plan";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--p" && i + 1 < argc) p = std::stod(argv[++i]);
    else if (a == "--n" && i + 1 < argc) cfg.n = std::stoi(argv[++i]);
    else if (a == "--wd" && i + 1 < argc) cfg.weight_decay = std::stof(argv[++i]);
    else if (a == "--lr" && i + 1 < argc) cfg.lr = std::stof(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = std::stoi(argv[++i]);
    else if (a == "--mode" && i + 1 < argc) mode = std::stoi(argv[++i]);
    else if (a == "--steps" && i + 1 < argc) cfg.total_steps = std::stoi(argv[++i]);
    else if (a == "--release" && i + 1 < argc) cfg.release_step = std::stoi(argv[++i]);
    else if (a == "--outdir" && i + 1 < argc) out_dir = argv[++i];
    else if (a == "--fast-probe") cfg.fast_probe = true;
  }

  std::filesystem::create_directories(out_dir);
  std::ostringstream fn;
  fn << out_dir << "/n" << cfg.n << "_p" << std::fixed << std::setprecision(2) << p
     << "_m" << mode << "_s" << seed << ".csv";
  auto t0 = std::chrono::steady_clock::now();
  run_single(cfg, p, seed, mode, fn.str());
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "Done n=" << cfg.n << " p=" << p << " mode=" << mode << " seed=" << seed
            << " in " << std::setprecision(2) << sec << "s\n";
  return 0;
}
