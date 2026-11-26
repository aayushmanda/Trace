// Blazing-fast float32 Apple Accelerate (cblas_sgemm / AMX) C++20 implementation of the
// unconstrained 27-block Transformer and surgical causal shortcut-calibration interventions:
//   Mode 0 (Natural):    Unmodified 27-block training from V(0) = 0
//   Mode 1 (Suppressed): In-distribution orthogonal logit projection \Pi_0^{(p)} (\hat c_{t,p} == 0)
//   Mode 2 (Calibrated): Pre-calibrated state-prior baseline +0.5 * c* * x_{t-1}(e_0 - e_1) from step 0 (unclamped)
//   Mode 3 (Restored):   Suppressed (\Pi_0^{(p)}) for step < release_step, then released cleanly!
//
// Includes double-precision finite-difference gradient check of \Pi_0^{(p)} at startup.
// Compile:
//   clang++ -O3 -std=c++20 -DACCELERATE_NEW_LAPACK -framework Accelerate run_causal_fast.cpp -o run_causal_fast

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include "cblas_shim.h"
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
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
  float init_scale = 0.02f;
  float lr = 1e-3f;
  float weight_decay = 0.01f;
  int batch_size = 256;
  int total_steps = 2400;
  int eval_every = 100;
  int release_step = 1000;
  int test_size = 1000;
  int n_threads = 4;
};

enum class Mode {
  NATURAL = 0,
  SUPPRESSED = 1,
  CALIBRATED = 2,      // External logit offset +0.5 c* x_{t-1}(e_0 - e_1) with V(0) = 0
  RESTORED = 3,        // Suppressed (\Pi_0^{(p)}) for step < release_step, then released
  WEIGHT_CALIBRATED = 4 // Readout weight V(0) initialized on state embedding difference to realize c* inside V!
};

class ThreadPool {
 public:
  explicit ThreadPool(int n) : n_threads_(n), stop_(false), gen_(0), active_(0) {
    for (int tid = 0; tid < n; ++tid) {
      workers_.emplace_back([this, tid]() { worker_loop(tid); });
    }
  }
  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
      gen_++;
    }
    cv_start_.notify_all();
    for (auto& w : workers_) w.join();
  }
  void run_chunks(const std::function<void(int)>& fn) {
    if (n_threads_ == 1) {
      fn(0);
      return;
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      fn_ = fn;
      active_ = n_threads_;
      gen_++;
    }
    cv_start_.notify_all();
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [this]() { return active_ == 0; });
  }
  int size() const { return n_threads_; }

 private:
  void worker_loop(int tid) {
    uint64_t my_gen = 0;
    while (true) {
      std::function<void(int)> local_fn;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_start_.wait(lk, [&]() { return gen_ != my_gen; });
        if (stop_) return;
        my_gen = gen_;
        local_fn = fn_;
      }
      local_fn(tid);
      {
        std::lock_guard<std::mutex> lk(mu_);
        if (--active_ == 0) cv_done_.notify_one();
      }
    }
  }
  int n_threads_;
  bool stop_;
  uint64_t gen_;
  int active_;
  std::function<void(int)> fn_;
  std::mutex mu_;
  std::condition_variable cv_start_, cv_done_;
  std::vector<std::thread> workers_;
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
  std::vector<float> q[2][3], k[2][3], v[2][3], alpha[2][3];
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
    logits.resize(ML * 2);
    dq.resize(ML * DK);
    dk.resize(ML * DK);
    dv.resize(ML * D);
  }
};

void forward_chunk(const Params& P, const Prompt* prompts, int M, ChunkBuffer& CB) {
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

      // Compute M * L * L causal scores via M batched sgemm_nt(L, L, DK) calls!
      for (int m = 0; m < M; ++m) {
        const float* qm = CB.q[l][h].data() + m * L * DK;
        const float* km = CB.k[l][h].data() + m * L * DK;
        const float* vm = CB.v[l][h].data() + m * L * D;
        float* Am = CB.alpha[l][h].data() + m * L * L;
        float* htm = CB.htilde[l].data() + m * L * D;

        sgemm_nt(L, L, DK, inv_sqrt_dk, qm, km, 0.0f, Am);
        for (int i = 0; i < L; ++i) {
          float* row = Am + i * L;
          float max_s = -1e30f;
          for (int j = 0; j <= i; ++j) {
            int lag = i - j;
            if (l == 1 && lag > 0) row[j] += beta[lag];
            if (row[j] > max_s) max_s = row[j];
          }
          float sum_exp = 0.0f;
          for (int j = 0; j <= i; ++j) {
            float e = std::exp(row[j] - max_s);
            row[j] = e;
            sum_exp += e;
          }
          float inv_sum = 1.0f / sum_exp;
          for (int j = 0; j <= i; ++j) row[j] *= inv_sum;
          for (int j = i + 1; j < L; ++j) row[j] = 0.0f;
        }
        sgemm_nn(L, D, L, 1.0f, Am, vm, 1.0f, htm);
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

  float dA[36 * 36]; // L <= 36
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

      for (int m = 0; m < M; ++m) {
        const float* Am = CB.alpha[l][h].data() + m * L * L;
        const float* dhtm = CB.dhtilde[l].data() + m * L * D;
        const float* qm = CB.q[l][h].data() + m * L * DK;
        const float* km = CB.k[l][h].data() + m * L * DK;
        const float* vm = CB.v[l][h].data() + m * L * D;
        float* dqm = CB.dq.data() + m * L * DK;
        float* dkm = CB.dk.data() + m * L * DK;
        float* dvm = CB.dv.data() + m * L * D;

        // dv = A^T * dhtilde (L x D); dA = dhtilde * v^T (L x L) via BLAS!
        sgemm_tn(L, D, L, 1.0f, Am, dhtm, 0.0f, dvm);
        sgemm_nt(L, L, D, 1.0f, dhtm, vm, 0.0f, dA);

        for (int i = 0; i < L; ++i) {
          float dot_adA = 0.0f;
          for (int j = 0; j <= i; ++j) dot_adA += Am[i * L + j] * dA[i * L + j];
          for (int j = 0; j <= i; ++j) {
            float ds = Am[i * L + j] * (dA[i * L + j] - dot_adA);
            int lag = i - j;
            if (l == 1 && lag > 0) dbeta[lag] += ds;
            dA[i * L + j] = ds * inv_sqrt_dk;
          }
          for (int j = i + 1; j < L; ++j) dA[i * L + j] = 0.0f;
        }
        // dq = dS_scaled * k (L x DK); dk = dS_scaled^T * q (L x DK) via BLAS!
        sgemm_nn(L, DK, L, 1.0f, dA, km, 0.0f, dqm);
        sgemm_tn(L, DK, L, 1.0f, dA, qm, 0.0f, dkm);
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

struct EvalStats {
  double step_loss = 0.0;
  double one_step_acc = 0.0;
  double rollout_acc = 0.0;
  double c_hat = 0.0;
  double lag_n_share = 0.0;
  double E_dist = 0.0;
  double E_true = 0.0;
};

EvalStats evaluate_model(
    ThreadPool& pool,
    const Config& cfg,
    const Params& P,
    const std::vector<Prompt>& test_set,
    bool suppress_c0,
    float add_c_offset,
    std::vector<ChunkBuffer>& thread_cb) {

  const int NT = pool.size();
  const int N = (int)test_set.size();
  const int n = cfg.n;
  const int L = P.seq_len;

  // 1. Forward pass across test_set + compute \hat c_{t,p} on test_set
  std::vector<std::vector<double>> th_chat(NT, std::vector<double>(n + 1, 0.0));
  pool.run_chunks([&](int tid) {
    int s = (tid * N) / NT, e = ((tid + 1) * N) / NT;
    int M = e - s;
    if (M <= 0) return;
    forward_chunk(P, test_set.data() + s, M, thread_cb[tid]);
    for (int m = 0; m < M; ++m) {
      for (int t = 1; t <= n; ++t) {
        int pos = n + 1 + t;
        double x_prev = (test_set[s + m].states[t - 1] == 0) ? 1.0 : -1.0;
        double dz = (double)thread_cb[tid].logits[(m * L + pos) * 2 + 0] -
                    (double)thread_cb[tid].logits[(m * L + pos) * 2 + 1] + 1.0 * add_c_offset * x_prev;
        th_chat[tid][t] += x_prev * dz / N;
      }
    }
  });

  std::vector<double> c_hat_t(n + 1, 0.0);
  for (int tid = 0; tid < NT; ++tid) {
    for (int t = 1; t <= n; ++t) c_hat_t[t] += th_chat[tid][t];
  }

  EvalStats st;
  for (int t = 2; t <= n; ++t) st.c_hat += (suppress_c0 ? 0.0 : c_hat_t[t]) / (n - 1);

  std::vector<double> th_loss(NT, 0.0);
  std::vector<long long> th_1step(NT, 0), th_roll(NT, 0);
  std::vector<std::array<double, 3>> th_head_lag(NT, {0.0, 0.0, 0.0});
  std::vector<std::vector<double>> th_corr(NT, std::vector<double>((n + 1) * (n + 1), 0.0));

  pool.run_chunks([&](int tid) {
    int s = (tid * N) / NT, e = ((tid + 1) * N) / NT;
    int M = e - s;
    if (M <= 0) return;
    ChunkBuffer& CB = thread_cb[tid];

    for (int m = 0; m < M; ++m) {
      const Prompt& pr = test_set[s + m];
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
        th_loss[tid] += -std::log(p_true + 1e-300) / ((double)N * n);
        if (((z0 >= z1) ? 0 : 1) == y) th_1step[tid]++;

        if (t >= 2) {
          double res_mag = 1.0 - p_true;
          double chi_t = pr.bits[t - 1] ? -1.0 : 1.0;
          for (int r = 1; r <= n; ++r) {
            double chi_r = pr.bits[r - 1] ? -1.0 : 1.0;
            th_corr[tid][t * (n + 1) + r] += (res_mag * chi_t * chi_r) / N;
          }
          for (int h = 0; h < 3; ++h) {
            const float* A = CB.alpha[1][h].data() + (m * L + pos) * L;
            double let_sum = 0.0;
            for (int r = 1; r <= n; ++r) let_sum += A[1 + r];
            double pi_star = (let_sum > 1e-12) ? (A[1 + t] / let_sum) : (1.0 / n);
            th_head_lag[tid][h] += pi_star / ((double)N * (n - 1));
          }
        }
      }
    }

    // Incremental greedy rollout
    std::vector<int> prev_s(M);
    std::vector<bool> ok(M, true);
    for (int m = 0; m < M; ++m) prev_s[m] = test_set[s + m].states[0];
    std::vector<float> p_hin(M * P.d_model), p_q(M * P.d_k), p_k(M * P.d_k), p_v(M * P.d_model);
    std::vector<float> p_u(M * P.d_mlp), p_r(M * P.d_mlp), p_ho(M * P.d_model), p_z(M * 2);

    for (int t = 1; t <= n; ++t) {
      int pos = n + 1 + t;
      if (t >= 2) {
        forward_chunk_single_pos(P, pos, prev_s.data(), M, CB, p_hin, p_q, p_k, p_v, p_u, p_r, p_ho, p_z);
      }
      for (int m = 0; m < M; ++m) {
        double x_prev = (prev_s[m] == 0) ? 1.0 : -1.0;
        double z0 = CB.logits[(m * L + pos) * 2 + 0] + 0.5 * add_c_offset * x_prev;
        double z1 = CB.logits[(m * L + pos) * 2 + 1] - 0.5 * add_c_offset * x_prev;
        if (suppress_c0) {
          double shift = -0.5 * c_hat_t[t] * x_prev;
          z0 += shift;
          z1 -= shift;
        }
        int pred = (z0 >= z1) ? 0 : 1;
        if (pred != test_set[s + m].states[t]) ok[m] = false;
        prev_s[m] = pred;
      }
    }
    for (int m = 0; m < M; ++m) if (ok[m]) th_roll[tid]++;
  });

  long long tot_1step = 0, tot_roll = 0;
  double head_lag[3] = {0.0, 0.0, 0.0};
  std::vector<double> corr((n + 1) * (n + 1), 0.0);
  for (int tid = 0; tid < NT; ++tid) {
    st.step_loss += th_loss[tid];
    tot_1step += th_1step[tid];
    tot_roll += th_roll[tid];
    for (int h = 0; h < 3; ++h) head_lag[h] += th_head_lag[tid][h];
    for (size_t k = 0; k < corr.size(); ++k) corr[k] += th_corr[tid][k];
  }
  st.one_step_acc = 100.0 * (double)tot_1step / ((double)N * n);
  st.rollout_acc = 100.0 * (double)tot_roll / (double)N;
  st.lag_n_share = std::max({head_lag[0], head_lag[1], head_lag[2]});

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

void run_single_condition(
    ThreadPool& pool,
    const Config& cfg,
    double p,
    int seed,
    Mode mode,
    const std::string& csv_path) {

  const float c_star = (float)std::log((1.0 - p) / p);
  Params P;
  P.allocate(cfg);
  P.init_random(seed, cfg.init_scale, true);
  if (mode == Mode::WEIGHT_CALIBRATED && std::abs(c_star) > 1e-7f) {
    // Initialize V(0) along v_state = 0.5 * (E_tok[0] - E_tok[1]) so (V h^{(0)})_0 - (V h^{(0)})_1 == c* x_{t-1} at step 0!
    const int D = cfg.d_model;
    const float* e0 = P.data.data() + P.off_E_tok + 0 * D;
    const float* e1 = P.data.data() + P.off_E_tok + 1 * D;
    double norm2 = 0.0;
    std::vector<double> vs(D);
    for (int d = 0; d < D; ++d) {
      vs[d] = 0.5 * ((double)e0[d] - (double)e1[d]);
      norm2 += vs[d] * vs[d];
    }
    if (norm2 > 1e-12) {
      for (int d = 0; d < D; ++d) {
        float w = (float)(0.5 * (double)c_star * vs[d] / norm2);
        P.data[P.off_V + 0 * D + d] = +w;
        P.data[P.off_V + 1 * D + d] = -w;
      }
    }
  }

  std::mt19937_64 test_rng(100000 + seed);
  std::vector<Prompt> test_set(cfg.test_size);
  for (int m = 0; m < cfg.test_size; ++m) test_set[m] = sample_prompt(test_rng, cfg.n, cfg.sigma, p);

  std::mt19937_64 data_rng(200000 + seed);

  std::vector<float> m_buf(P.data.size(), 0.0f), v_buf(P.data.size(), 0.0f), grad(P.data.size(), 0.0f);
  const float b1 = 0.9f, b2 = 0.999f, eps_adam = 1e-8f;

  const int NT = pool.size();
  const int n = cfg.n;
  const int L = P.seq_len;
  const int B = cfg.batch_size;
  int max_per_thread = std::max((B + NT - 1) / NT, (cfg.test_size + NT - 1) / NT) + 2;

  std::vector<ChunkBuffer> train_cb(NT), eval_cb(NT);
  for (int tid = 0; tid < NT; ++tid) {
    train_cb[tid].init(cfg, max_per_thread);
    eval_cb[tid].init(cfg, max_per_thread);
  }
  std::vector<std::vector<float>> thread_grads(NT, std::vector<float>(P.data.size(), 0.0f));

  std::ofstream ofs(csv_path);
  ofs << "step,loss,loss_bits,one_step_acc,rollout_acc,c_sym,lag_n_share,E_dist,E_true\n";

  for (int step = 0; step <= cfg.total_steps; ++step) {
    bool suppress_c0 = false;
    float add_c_offset = 0.0f;
    if (mode == Mode::SUPPRESSED) {
      suppress_c0 = true;
    } else if (mode == Mode::CALIBRATED) {
      add_c_offset = c_star;
    } else if (mode == Mode::RESTORED) {
      if (step < cfg.release_step) {
        suppress_c0 = true;
      } else {
        suppress_c0 = false;
        add_c_offset = 0.0f; // cleanly release suppression so natural shortcut + routing proceed!
      }
    }

    if (step % cfg.eval_every == 0) {
      EvalStats st = evaluate_model(pool, cfg, P, test_set, suppress_c0, add_c_offset, eval_cb);
      double loss_bits = st.step_loss / std::log(2.0);
      ofs << step << "," << std::fixed << std::setprecision(6) << st.step_loss << "," << loss_bits << ","
          << std::setprecision(2) << st.one_step_acc << "," << st.rollout_acc << ","
          << std::setprecision(4) << st.c_hat << "," << st.lag_n_share << ","
          << st.E_dist << "," << st.E_true << "\n";
      ofs.flush();
    }

    if (step == cfg.total_steps) break;

    std::vector<Prompt> batch(B);
    for (int m = 0; m < B; ++m) batch[m] = sample_prompt(data_rng, n, cfg.sigma, p);

    pool.run_chunks([&](int tid) {
      int s = (tid * B) / NT, e = ((tid + 1) * B) / NT;
      forward_chunk(P, batch.data() + s, e - s, train_cb[tid]);
    });

    // Compute batch mean state-aligned logit \hat c_{t,p} if suppress_c0
    std::vector<double> c_hat_batch(n + 1, 0.0);
    if (suppress_c0) {
      for (int tid = 0; tid < NT; ++tid) {
        int s = (tid * B) / NT, e = ((tid + 1) * B) / NT;
        for (int m = 0; m < e - s; ++m) {
          int gm = s + m;
          for (int t = 1; t <= n; ++t) {
            int pos = n + 1 + t;
            double x_prev = (batch[gm].states[t - 1] == 0) ? 1.0 : -1.0;
            double dz = (double)train_cb[tid].logits[(m * L + pos) * 2 + 0] -
                        (double)train_cb[tid].logits[(m * L + pos) * 2 + 1];
            c_hat_batch[t] += x_prev * dz / B;
          }
        }
      }
    }

    // Compute loss gradients wrt intervened logits AND project gradient orthogonal to state-only subspace if suppress_c0!
    // Exact chain rule for z_t^supp(m) = z_t(m) - 0.5 * (\frac{1}{B}\sum_{m'} x'_{t-1}(z_{t,0}(m') - z_{t,1}(m'))) * x_{t-1}(m) (e_0 - e_1):
    // Let g_t(m) = dL / dz_t^supp(m), with g_{t,0}(m) + g_{t,1}(m) == 0, so g_t(m) = 0.5 * d_t(m) * (e_0 - e_1) where d_t(m) = g_{t,0}(m) - g_{t,1}(m).
    // Then dL / dz_t(m') = g_t(m') - 0.5 * (\frac{1}{B}\sum_m x_{t-1}(m) d_t(m)) * x_{t-1}(m') (e_0 - e_1)!!
    std::vector<float> train_dlogits(B * L * 2, 0.0f);
    std::vector<double> mean_x_d(n + 1, 0.0);
    const double inv_Bn = 1.0 / ((double)B * n);

    for (int tid = 0; tid < NT; ++tid) {
      int s = (tid * B) / NT, e = ((tid + 1) * B) / NT;
      for (int m = 0; m < e - s; ++m) {
        int gm = s + m;
        for (int t = 1; t <= n; ++t) {
          int pos = n + 1 + t;
          double x_prev = (batch[gm].states[t - 1] == 0) ? 1.0 : -1.0;
          double z0 = (double)train_cb[tid].logits[(m * L + pos) * 2 + 0] + 0.5 * add_c_offset * x_prev;
          double z1 = (double)train_cb[tid].logits[(m * L + pos) * 2 + 1] - 0.5 * add_c_offset * x_prev;
          if (suppress_c0) {
            double shift = -0.5 * c_hat_batch[t] * x_prev;
            z0 += shift;
            z1 -= shift;
          }
          double mx = std::max(z0, z1);
          double e0 = std::exp(z0 - mx), e1 = std::exp(z1 - mx);
          double p0 = e0 / (e0 + e1), p1 = e1 / (e0 + e1);
          int y = batch[gm].states[t];
          double g0 = (p0 - (y == 0 ? 1.0 : 0.0)) * inv_Bn;
          double g1 = (p1 - (y == 1 ? 1.0 : 0.0)) * inv_Bn;
          train_dlogits[(gm * L + pos) * 2 + 0] = (float)g0;
          train_dlogits[(gm * L + pos) * 2 + 1] = (float)g1;
          if (suppress_c0) {
            mean_x_d[t] += (x_prev * (g0 - g1)) / B; // exact 1/B from d(c_hat_batch)/dz(gm)!
          }
        }
      }
    }

    if (suppress_c0) {
      for (int gm = 0; gm < B; ++gm) {
        for (int t = 1; t <= n; ++t) {
          int pos = n + 1 + t;
          double x_prev = (batch[gm].states[t - 1] == 0) ? 1.0 : -1.0;
          // Exact orthogonal projection + mild nullspace damping (0.1 * c_hat_batch) so raw c_hat stays ~ 0
          float proj_sub = (float)(0.5 * (mean_x_d[t] - (0.1 / ((double)B * n)) * c_hat_batch[t]) * x_prev);
          train_dlogits[(gm * L + pos) * 2 + 0] -= proj_sub;
          train_dlogits[(gm * L + pos) * 2 + 1] += proj_sub;
        }
      }
    }

    pool.run_chunks([&](int tid) {
      std::fill(thread_grads[tid].begin(), thread_grads[tid].end(), 0.0f);
      int s = (tid * B) / NT, e = ((tid + 1) * B) / NT;
      backward_chunk(P, e - s, train_cb[tid], train_dlogits.data() + s * L * 2, thread_grads[tid].data());
    });

    std::fill(grad.begin(), grad.end(), 0.0f);
    for (int tid = 0; tid < NT; ++tid) {
      for (size_t i = 0; i < grad.size(); ++i) grad[i] += thread_grads[tid][i];
    }

    float bc1 = 1.0f - std::pow(b1, step + 1);
    float bc2 = 1.0f - std::pow(b2, step + 1);
    for (const auto& b : P.blocks) {
      if (b.name.rfind("B1", 0) == 0 && b.is_lag_bias) continue;
      for (size_t i = 0; i < b.size; ++i) {
        if (b.is_lag_bias && i == 0) continue;
        size_t idx = b.offset + i;
        float g = grad[idx];
        m_buf[idx] = b1 * m_buf[idx] + (1.0f - b1) * g;
        v_buf[idx] = b2 * v_buf[idx] + (1.0f - b2) * g * g;
        float mh = m_buf[idx] / bc1;
        float vh = v_buf[idx] / bc2;
        if (!b.no_wd) P.data[idx] *= (1.0f - cfg.lr * cfg.weight_decay);
        P.data[idx] -= cfg.lr * mh / (std::sqrt(vh) + eps_adam);
      }
    }
  }
}

int main(int argc, char** argv) {
  setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
  Config cfg;
  double p = 0.25;
  int seed = 42;
  int mode_int = 0;
  std::string out_dir = "results_fast";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--p" && i + 1 < argc) p = std::stod(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = std::stoi(argv[++i]);
    else if (a == "--mode" && i + 1 < argc) mode_int = std::stoi(argv[++i]);
    else if (a == "--steps" && i + 1 < argc) cfg.total_steps = std::stoi(argv[++i]);
    else if (a == "--release" && i + 1 < argc) cfg.release_step = std::stoi(argv[++i]);
    else if (a == "--eval-every" && i + 1 < argc) cfg.eval_every = std::stoi(argv[++i]);
    else if (a == "--threads" && i + 1 < argc) cfg.n_threads = std::stoi(argv[++i]);
    else if (a == "--outdir" && i + 1 < argc) out_dir = argv[++i];
  }

  std::filesystem::create_directories(out_dir);
  ThreadPool pool(cfg.n_threads);

  auto t0 = std::chrono::steady_clock::now();
  std::ostringstream fn;
  fn << out_dir << "/p" << std::fixed << std::setprecision(2) << p << "_m" << mode_int << "_s" << seed << ".csv";
  run_single_condition(pool, cfg, p, seed, static_cast<Mode>(mode_int), fn.str());
  auto t1 = std::chrono::steady_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "Finished p=" << p << " mode=" << mode_int << " seed=" << seed
            << " (" << cfg.total_steps << " steps) in " << std::setprecision(2) << sec << "s -> " << fn.str() << "\n";
  return 0;
}
