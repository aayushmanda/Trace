// Ultra-fast contiguous-batch GEMM C++20 implementation of the unconstrained 27-block Transformer
// for causal shortcut-calibration experiments (Natural, Suppressed c=0, Calibrated c=c*, Restored).
//
// Compile:
//   clang++ -O3 -std=c++20 -DACCELERATE_NEW_LAPACK -framework Accelerate run_causal_sweep.cpp -o run_causal_sweep

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
  double init_scale = 0.02;
  double lr = 1e-3;
  double weight_decay = 0.01;
  int batch_size = 256;
  int sym_quads = 16; // 64 antithetic balanced probe sequences
  int total_steps = 2400;
  int eval_every = 100;
  int release_step = 1000;
  int test_size = 500;
  int n_threads = 8;
};

enum class Mode {
  NATURAL = 0,     // Unmodified 27-block Transformer training
  SUPPRESSED = 1,  // Orthogonal logit projection \bar c_t -> 0
  CALIBRATED = 2,  // Orthogonal logit projection \bar c_t -> c* = log((1-p)/p)
  RESTORED = 3     // Suppress \bar c_t -> 0 for step < release_step, then release to natural (or c*)
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
  std::vector<double> data;
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
    data.assign(cur, 0.0);
  }

  void init_random(uint64_t seed, double scale, bool zero_V = true) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, scale);
    for (const auto& b : blocks) {
      if (b.is_lag_bias || (zero_V && b.name == "V")) {
        std::fill(data.begin() + b.offset, data.begin() + b.offset + b.size, 0.0);
      } else {
        for (size_t i = 0; i < b.size; ++i) data[b.offset + i] = nd(rng);
      }
    }
  }
};

inline void gemm_nt(int M, int N, int K, double alpha, const double* A, const double* B, double beta, double* C) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, N);
}
inline void gemm_nn(int M, int N, int K, double alpha, const double* A, const double* B, double beta, double* C) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
}
inline void gemm_tn(int M, int N, int K, double alpha, const double* A, const double* B, double beta, double* C) {
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C, N);
}

// Contiguous chunk buffer holding `max_M` sequences (M * L rows) so every linear layer is ONE large GEMM!
struct ChunkBuffer {
  int max_M, L, D, DK, DM;
  std::vector<int> tokens; // max_M * L
  std::vector<double> h0;
  std::vector<double> q[2][3], k[2][3], v[2][3], alpha[2][3]; // alpha: max_M * L * L
  std::vector<double> htilde[2], u_mlp[2], r_mlp[2], h_out[2];
  std::vector<double> logits; // max_M * L * 2

  // Backward scratch
  std::vector<double> dh_out[2], dhtilde[2], du_mlp[2], dr_mlp[2];
  std::vector<double> dq, dk, dv;

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
        alpha[l][h].assign(max_M * L * L, 0.0);
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
  const double inv_sqrt_dk = 1.0 / std::sqrt((double)DK);

  const double* E_tok = P.data.data() + P.off_E_tok;
  const double* E_pos = P.data.data() + P.off_E_pos;
  for (int m = 0; m < M; ++m) {
    const int* ptok = prompts[m].tokens.data();
    for (int i = 0; i < L; ++i) {
      int tok = ptok[i];
      CB.tokens[m * L + i] = tok;
      const double* et = E_tok + tok * D;
      const double* ep = E_pos + i * D;
      double* dst = CB.h0.data() + (m * L + i) * D;
      for (int d = 0; d < D; ++d) dst[d] = et[d] + ep[d];
    }
  }

  double scores[64];
  for (int l = 0; l < 2; ++l) {
    const double* hin = (l == 0) ? CB.h0.data() : CB.h_out[0].data();
    std::memcpy(CB.htilde[l].data(), hin, sizeof(double) * ML * D);

    for (int h = 0; h < 3; ++h) {
      const double* WQ = P.data.data() + P.off_WQ[l][h];
      const double* WK = P.data.data() + P.off_WK[l][h];
      const double* WV = P.data.data() + P.off_WV[l][h];
      const double* beta = P.data.data() + P.off_beta[l][h];

      // Single batched GEMMs across all M * L tokens!
      gemm_nt(ML, DK, D, 1.0, hin, WQ, 0.0, CB.q[l][h].data());
      gemm_nt(ML, DK, D, 1.0, hin, WK, 0.0, CB.k[l][h].data());
      gemm_nt(ML, D, D, 1.0, hin, WV, 0.0, CB.v[l][h].data());

      for (int m = 0; m < M; ++m) {
        const double* qm = CB.q[l][h].data() + m * L * DK;
        const double* km = CB.k[l][h].data() + m * L * DK;
        const double* vm = CB.v[l][h].data() + m * L * D;
        double* Am = CB.alpha[l][h].data() + m * L * L;
        double* htm = CB.htilde[l].data() + m * L * D;

        for (int i = 0; i < L; ++i) {
          const double* qi = qm + i * DK;
          double max_s = -1e300;
          for (int j = 0; j <= i; ++j) {
            const double* kj = km + j * DK;
            double dot = 0.0;
            for (int d = 0; d < DK; ++d) dot += qi[d] * kj[d];
            int lag = i - j;
            double b_lag = (l == 1 && lag > 0) ? beta[lag] : 0.0;
            double s = b_lag + dot * inv_sqrt_dk;
            scores[j] = s;
            if (s > max_s) max_s = s;
          }
          double sum_exp = 0.0;
          for (int j = 0; j <= i; ++j) {
            double e = std::exp(scores[j] - max_s);
            Am[i * L + j] = e;
            sum_exp += e;
          }
          double inv_sum = 1.0 / sum_exp;
          double* hti = htm + i * D;
          for (int j = 0; j <= i; ++j) {
            double w = (Am[i * L + j] *= inv_sum);
            const double* vj = vm + j * D;
            for (int d = 0; d < D; ++d) hti[d] += w * vj[d];
          }
        }
      }
    }

    const double* Wup = P.data.data() + P.off_Wup[l];
    const double* bup = P.data.data() + P.off_bup[l];
    const double* Wdown = P.data.data() + P.off_Wdown[l];
    gemm_nt(ML, DM, D, 1.0, CB.htilde[l].data(), Wup, 0.0, CB.u_mlp[l].data());
    for (int row = 0; row < ML; ++row) {
      double* ui = CB.u_mlp[l].data() + row * DM;
      double* ri = CB.r_mlp[l].data() + row * DM;
      for (int c = 0; c < DM; ++c) {
        double val = ui[c] + bup[c];
        ui[c] = val;
        ri[c] = (val > 0.0) ? val : 0.0;
      }
    }
    std::memcpy(CB.h_out[l].data(), CB.htilde[l].data(), sizeof(double) * ML * D);
    gemm_nt(ML, D, DM, 1.0, CB.r_mlp[l].data(), Wdown, 1.0, CB.h_out[l].data());
  }

  const double* V = P.data.data() + P.off_V;
  gemm_nt(ML, 2, D, 1.0, CB.h_out[1].data(), V, 0.0, CB.logits.data());
}

void backward_chunk(const Params& P, int M, ChunkBuffer& CB, const double* d_logits, double* grad) {
  if (M <= 0) return;
  const int L = P.seq_len, ML = M * L;
  const int D = P.d_model, DK = P.d_k, DM = P.d_mlp;
  const double inv_sqrt_dk = 1.0 / std::sqrt((double)DK);

  const double* V = P.data.data() + P.off_V;
  gemm_tn(2, D, ML, 1.0, d_logits, CB.h_out[1].data(), 1.0, grad + P.off_V);
  gemm_nn(ML, D, 2, 1.0, d_logits, V, 0.0, CB.dh_out[1].data());

  for (int l = 1; l >= 0; --l) {
    const double* hin = (l == 0) ? CB.h0.data() : CB.h_out[0].data();
    const double* Wup = P.data.data() + P.off_Wup[l];
    const double* Wdown = P.data.data() + P.off_Wdown[l];

    std::memcpy(CB.dhtilde[l].data(), CB.dh_out[l].data(), sizeof(double) * ML * D);
    gemm_tn(D, DM, ML, 1.0, CB.dh_out[l].data(), CB.r_mlp[l].data(), 1.0, grad + P.off_Wdown[l]);
    gemm_nn(ML, DM, D, 1.0, CB.dh_out[l].data(), Wdown, 0.0, CB.dr_mlp[l].data());

    double* dbup = grad + P.off_bup[l];
    for (int row = 0; row < ML; ++row) {
      const double* ui = CB.u_mlp[l].data() + row * DM;
      const double* dri = CB.dr_mlp[l].data() + row * DM;
      double* dui = CB.du_mlp[l].data() + row * DM;
      for (int c = 0; c < DM; ++c) {
        double g = (ui[c] > 0.0) ? dri[c] : 0.0;
        dui[c] = g;
        dbup[c] += g;
      }
    }
    gemm_tn(DM, D, ML, 1.0, CB.du_mlp[l].data(), CB.htilde[l].data(), 1.0, grad + P.off_Wup[l]);
    gemm_nn(ML, D, DM, 1.0, CB.du_mlp[l].data(), Wup, 1.0, CB.dhtilde[l].data());

    double* dhin = (l == 1) ? CB.dh_out[0].data() : CB.dh_out[1].data();
    std::memcpy(dhin, CB.dhtilde[l].data(), sizeof(double) * ML * D);

    for (int h = 0; h < 3; ++h) {
      const double* WQ = P.data.data() + P.off_WQ[l][h];
      const double* WK = P.data.data() + P.off_WK[l][h];
      const double* WV = P.data.data() + P.off_WV[l][h];
      double* dbeta = grad + P.off_beta[l][h];

      std::fill(CB.dv.begin(), CB.dv.begin() + ML * D, 0.0);
      std::fill(CB.dq.begin(), CB.dq.begin() + ML * DK, 0.0);
      std::fill(CB.dk.begin(), CB.dk.begin() + ML * DK, 0.0);

      for (int m = 0; m < M; ++m) {
        const double* Am = CB.alpha[l][h].data() + m * L * L;
        const double* dhtm = CB.dhtilde[l].data() + m * L * D;
        const double* qm = CB.q[l][h].data() + m * L * DK;
        const double* km = CB.k[l][h].data() + m * L * DK;
        const double* vm = CB.v[l][h].data() + m * L * D;
        double* dqm = CB.dq.data() + m * L * DK;
        double* dkm = CB.dk.data() + m * L * DK;
        double* dvm = CB.dv.data() + m * L * D;

        for (int i = 0; i < L; ++i) {
          const double* dhti = dhtm + i * D;
          double dot_adA = 0.0;
          double dAi[64];
          for (int j = 0; j <= i; ++j) {
            const double* vj = vm + j * D;
            double da = 0.0;
            for (int d = 0; d < D; ++d) da += dhti[d] * vj[d];
            dAi[j] = da;
            double aij = Am[i * L + j];
            dot_adA += aij * da;
            double* dvj = dvm + j * D;
            for (int d = 0; d < D; ++d) dvj[d] += aij * dhti[d];
          }
          const double* qi = qm + i * DK;
          double* dqi = dqm + i * DK;
          for (int j = 0; j <= i; ++j) {
            double ds = Am[i * L + j] * (dAi[j] - dot_adA);
            int lag = i - j;
            if (l == 1 && lag > 0) dbeta[lag] += ds;
            double ds_scaled = ds * inv_sqrt_dk;
            const double* kj = km + j * DK;
            double* dkj = dkm + j * DK;
            for (int d = 0; d < DK; ++d) {
              dqi[d] += ds_scaled * kj[d];
              dkj[d] += ds_scaled * qi[d];
            }
          }
        }
      }

      gemm_tn(DK, D, ML, 1.0, CB.dq.data(), hin, 1.0, grad + P.off_WQ[l][h]);
      gemm_tn(DK, D, ML, 1.0, CB.dk.data(), hin, 1.0, grad + P.off_WK[l][h]);
      gemm_tn(D, D, ML, 1.0, CB.dv.data(), hin, 1.0, grad + P.off_WV[l][h]);
      gemm_nn(ML, D, DK, 1.0, CB.dq.data(), WQ, 1.0, dhin);
      gemm_nn(ML, D, DK, 1.0, CB.dk.data(), WK, 1.0, dhin);
      gemm_nn(ML, D, D, 1.0, CB.dv.data(), WV, 1.0, dhin);
    }
  }

  const double* dh0 = CB.dh_out[1].data();
  double* dE_tok = grad + P.off_E_tok;
  double* dE_pos = grad + P.off_E_pos;
  for (int m = 0; m < M; ++m) {
    for (int i = 0; i < L; ++i) {
      int row = m * L + i;
      const double* g = dh0 + row * D;
      double* dt = dE_tok + CB.tokens[row] * D;
      double* dp = dE_pos + i * D;
      for (int d = 0; d < D; ++d) {
        dt[d] += g[d];
        dp[d] += g[d];
      }
    }
  }
}

// Incremental single-step causal forward across M sequences simultaneously at position `pos`!
void forward_chunk_single_pos(const Params& P, int pos, const int* step_toks, int M, ChunkBuffer& CB,
                              std::vector<double>& pos_hin, std::vector<double>& pos_q,
                              std::vector<double>& pos_k, std::vector<double>& pos_v,
                              std::vector<double>& pos_u, std::vector<double>& pos_r,
                              std::vector<double>& pos_ho, std::vector<double>& pos_z) {
  const int L = P.seq_len, D = P.d_model, DK = P.d_k, DM = P.d_mlp;
  const double inv_sqrt_dk = 1.0 / std::sqrt((double)DK);

  const double* E_tok = P.data.data() + P.off_E_tok;
  const double* ep = P.data.data() + P.off_E_pos + pos * D;
  for (int m = 0; m < M; ++m) {
    int tok = step_toks[m];
    CB.tokens[m * L + pos] = tok;
    const double* et = E_tok + tok * D;
    double* dst = CB.h0.data() + (m * L + pos) * D;
    for (int d = 0; d < D; ++d) {
      double val = et[d] + ep[d];
      dst[d] = val;
      pos_hin[m * D + d] = val;
    }
  }

  double scores[64];
  for (int l = 0; l < 2; ++l) {
    if (l == 1) {
      for (int m = 0; m < M; ++m) {
        std::memcpy(pos_hin.data() + m * D, CB.h_out[0].data() + (m * L + pos) * D, sizeof(double) * D);
      }
    }
    // pos_ho temporarily accumulates htilde[l] at `pos`
    std::memcpy(pos_ho.data(), pos_hin.data(), sizeof(double) * M * D);

    for (int h = 0; h < 3; ++h) {
      const double* WQ = P.data.data() + P.off_WQ[l][h];
      const double* WK = P.data.data() + P.off_WK[l][h];
      const double* WV = P.data.data() + P.off_WV[l][h];
      const double* beta = P.data.data() + P.off_beta[l][h];
      gemm_nt(M, DK, D, 1.0, pos_hin.data(), WQ, 0.0, pos_q.data());
      gemm_nt(M, DK, D, 1.0, pos_hin.data(), WK, 0.0, pos_k.data());
      gemm_nt(M, D, D, 1.0, pos_hin.data(), WV, 0.0, pos_v.data());

      for (int m = 0; m < M; ++m) {
        std::memcpy(CB.k[l][h].data() + (m * L + pos) * DK, pos_k.data() + m * DK, sizeof(double) * DK);
        std::memcpy(CB.v[l][h].data() + (m * L + pos) * D, pos_v.data() + m * D, sizeof(double) * D);
        const double* qi = pos_q.data() + m * DK;
        const double* km = CB.k[l][h].data() + m * L * DK;
        const double* vm = CB.v[l][h].data() + m * L * D;
        double max_s = -1e300;
        for (int j = 0; j <= pos; ++j) {
          const double* kj = km + j * DK;
          double dot = 0.0;
          for (int d = 0; d < DK; ++d) dot += qi[d] * kj[d];
          int lag = pos - j;
          double b_lag = (l == 1 && lag > 0) ? beta[lag] : 0.0;
          double s = b_lag + dot * inv_sqrt_dk;
          scores[j] = s;
          if (s > max_s) max_s = s;
        }
        double sum_exp = 0.0;
        for (int j = 0; j <= pos; ++j) {
          scores[j] = std::exp(scores[j] - max_s);
          sum_exp += scores[j];
        }
        double inv_sum = 1.0 / sum_exp;
        double* hti = pos_ho.data() + m * D;
        for (int j = 0; j <= pos; ++j) {
          double w = scores[j] * inv_sum;
          const double* vj = vm + j * D;
          for (int d = 0; d < D; ++d) hti[d] += w * vj[d];
        }
      }
    }

    const double* Wup = P.data.data() + P.off_Wup[l];
    const double* bup = P.data.data() + P.off_bup[l];
    const double* Wdown = P.data.data() + P.off_Wdown[l];
    gemm_nt(M, DM, D, 1.0, pos_ho.data(), Wup, 0.0, pos_u.data());
    for (int m = 0; m < M; ++m) {
      double* ui = pos_u.data() + m * DM;
      double* ri = pos_r.data() + m * DM;
      for (int c = 0; c < DM; ++c) {
        double val = ui[c] + bup[c];
        ri[c] = (val > 0.0) ? val : 0.0;
      }
    }
    gemm_nt(M, D, DM, 1.0, pos_r.data(), Wdown, 1.0, pos_ho.data());
    for (int m = 0; m < M; ++m) {
      std::memcpy(CB.h_out[l].data() + (m * L + pos) * D, pos_ho.data() + m * D, sizeof(double) * D);
    }
  }
  const double* V = P.data.data() + P.off_V;
  gemm_nt(M, 2, D, 1.0, pos_ho.data(), V, 0.0, pos_z.data());
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

void sample_antithetic_probe(std::mt19937_64& rng, int n, int sigma, int n_quads, std::vector<Prompt>& out) {
  out.clear();
  out.reserve(n_quads * 4);
  std::uniform_int_distribution<int> bit_dist(0, 1);
  std::uniform_int_distribution<int> q_dist(0, sigma - 1);
  std::uniform_int_distribution<int> nonmatch_dist(0, sigma - 2);
  for (int q = 0; q < n_quads; ++q) {
    int w_star = q_dist(rng);
    std::vector<int> base_bits(n), nonmatch_tok(n);
    for (int t = 0; t < n; ++t) {
      base_bits[t] = bit_dist(rng);
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
          pr.tokens[1 + t] = 2 + (bt ? w_star : nonmatch_tok[t - 1]);
        }
        pr.tokens[n + 2] = 2 + sigma;
        for (int t = 2; t <= n; ++t) pr.tokens[n + 1 + t] = pr.states[t - 1];
        out.push_back(std::move(pr));
      }
    }
  }
}

struct EvalStats {
  double step_loss = 0.0;
  double one_step_acc = 0.0;
  double rollout_acc = 0.0;
  double c_sym = 0.0;
  double lag_n_share = 0.0;
  double E_dist = 0.0;
  double E_true = 0.0;
};

EvalStats evaluate_chunked(
    ThreadPool& pool,
    const Config& cfg,
    const Params& P,
    const std::vector<Prompt>& test_set,
    const std::vector<Prompt>& eval_sym,
    bool project_shortcut,
    double c_target,
    std::vector<ChunkBuffer>& thread_cb) {

  const int NT = pool.size();
  const int N = (int)test_set.size();
  const int Bsym = (int)eval_sym.size();
  const int n = cfg.n;
  const int L = P.seq_len;

  std::vector<std::vector<double>> th_cbar(NT, std::vector<double>(n + 1, 0.0));
  pool.run_chunks([&](int tid) {
    int s = (tid * Bsym) / NT;
    int e = ((tid + 1) * Bsym) / NT;
    int M = e - s;
    if (M <= 0) return;
    forward_chunk(P, eval_sym.data() + s, M, thread_cb[tid]);
    for (int m = 0; m < M; ++m) {
      for (int t = 1; t <= n; ++t) {
        int pos = n + 1 + t;
        double x_prev = (eval_sym[s + m].states[t - 1] == 0) ? 1.0 : -1.0;
        double dz = thread_cb[tid].logits[(m * L + pos) * 2 + 0] - thread_cb[tid].logits[(m * L + pos) * 2 + 1];
        th_cbar[tid][t] += x_prev * dz / Bsym;
      }
    }
  });

  std::vector<double> c_bar(n + 1, 0.0);
  for (int tid = 0; tid < NT; ++tid) {
    for (int t = 1; t <= n; ++t) c_bar[t] += th_cbar[tid][t];
  }

  EvalStats st;
  for (int t = 2; t <= n; ++t) st.c_sym += (project_shortcut ? c_target : c_bar[t]) / (n - 1);

  std::vector<double> th_loss(NT, 0.0);
  std::vector<long long> th_1step(NT, 0), th_roll(NT, 0);
  std::vector<std::array<double, 3>> th_head_lag(NT, {0.0, 0.0, 0.0});
  std::vector<std::vector<double>> th_corr(NT, std::vector<double>((n + 1) * (n + 1), 0.0));

  pool.run_chunks([&](int tid) {
    int s = (tid * N) / NT;
    int e = ((tid + 1) * N) / NT;
    int M = e - s;
    if (M <= 0) return;
    ChunkBuffer& CB = thread_cb[tid];
    forward_chunk(P, test_set.data() + s, M, CB);

    for (int m = 0; m < M; ++m) {
      const Prompt& pr = test_set[s + m];
      for (int t = 1; t <= n; ++t) {
        int pos = n + 1 + t;
        double x_prev = (pr.states[t - 1] == 0) ? 1.0 : -1.0;
        double z0 = CB.logits[(m * L + pos) * 2 + 0];
        double z1 = CB.logits[(m * L + pos) * 2 + 1];
        if (project_shortcut) {
          double shift = 0.5 * (c_target - c_bar[t]) * x_prev;
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
            const double* A = CB.alpha[1][h].data() + (m * L + pos) * L;
            double let_sum = 0.0;
            for (int r = 1; r <= n; ++r) let_sum += A[1 + r];
            double pi_star = (let_sum > 1e-12) ? (A[1 + t] / let_sum) : (1.0 / n);
            th_head_lag[tid][h] += pi_star / ((double)N * (n - 1));
          }
        }
      }
    }

    // Batched incremental greedy rollout across all M test prompts simultaneously!
    std::vector<int> prev_s(M);
    std::vector<bool> ok(M, true);
    for (int m = 0; m < M; ++m) prev_s[m] = test_set[s + m].states[0];
    std::vector<double> p_hin(M * P.d_model), p_q(M * P.d_k), p_k(M * P.d_k), p_v(M * P.d_model);
    std::vector<double> p_u(M * P.d_mlp), p_r(M * P.d_mlp), p_ho(M * P.d_model), p_z(M * 2);

    for (int t = 1; t <= n; ++t) {
      int pos = n + 1 + t;
      if (t >= 2) {
        forward_chunk_single_pos(P, pos, prev_s.data(), M, CB, p_hin, p_q, p_k, p_v, p_u, p_r, p_ho, p_z);
      }
      for (int m = 0; m < M; ++m) {
        double x_prev = (prev_s[m] == 0) ? 1.0 : -1.0;
        double z0 = CB.logits[(m * L + pos) * 2 + 0];
        double z1 = CB.logits[(m * L + pos) * 2 + 1];
        if (project_shortcut) {
          double shift = 0.5 * (c_target - c_bar[t]) * x_prev;
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

  const double c_star = std::log((1.0 - p) / p);
  Params P;
  P.allocate(cfg);
  P.init_random(seed, cfg.init_scale, true);

  std::mt19937_64 test_rng(100000 + seed);
  std::vector<Prompt> test_set(cfg.test_size);
  for (int m = 0; m < cfg.test_size; ++m) test_set[m] = sample_prompt(test_rng, cfg.n, cfg.sigma, p);
  std::vector<Prompt> eval_sym;
  sample_antithetic_probe(test_rng, cfg.n, cfg.sigma, 64, eval_sym);

  std::mt19937_64 data_rng(200000 + seed);
  std::mt19937_64 sym_rng(300000 + seed);

  std::vector<double> m_buf(P.data.size(), 0.0), v_buf(P.data.size(), 0.0), grad(P.data.size(), 0.0);
  const double b1 = 0.9, b2 = 0.999, eps_adam = 1e-8;

  const int NT = pool.size();
  const int n = cfg.n;
  const int L = P.seq_len;
  const int B = cfg.batch_size;
  const int Bsym = cfg.sym_quads * 4;
  int max_per_thread = std::max({(B + NT - 1) / NT, (cfg.test_size + NT - 1) / NT, (256 + NT - 1) / NT}) + 2;

  std::vector<ChunkBuffer> train_cb(NT), sym_cb(NT), eval_cb(NT);
  for (int tid = 0; tid < NT; ++tid) {
    train_cb[tid].init(cfg, max_per_thread);
    sym_cb[tid].init(cfg, (Bsym + NT - 1) / NT + 2);
    eval_cb[tid].init(cfg, max_per_thread);
  }
  std::vector<std::vector<double>> thread_grads(NT, std::vector<double>(P.data.size(), 0.0));

  std::ofstream ofs(csv_path);
  ofs << "step,loss,loss_bits,one_step_acc,rollout_acc,c_sym,lag_n_share,E_dist,E_true\n";

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
      EvalStats st = evaluate_chunked(pool, cfg, P, test_set, eval_sym, proj, c_targ, eval_cb);
      double loss_bits = st.step_loss / std::log(2.0);
      ofs << step << "," << std::fixed << std::setprecision(6) << st.step_loss << "," << loss_bits << ","
          << std::setprecision(2) << st.one_step_acc << "," << st.rollout_acc << ","
          << std::setprecision(4) << st.c_sym << "," << st.lag_n_share << ","
          << st.E_dist << "," << st.E_true << "\n";
      ofs.flush();
    }

    if (step == cfg.total_steps) break;

    std::vector<Prompt> batch(B), sym_batch;
    for (int m = 0; m < B; ++m) batch[m] = sample_prompt(data_rng, n, cfg.sigma, p);
    sample_antithetic_probe(sym_rng, n, cfg.sigma, cfg.sym_quads, sym_batch);

    pool.run_chunks([&](int tid) {
      int s = (tid * B) / NT, e = ((tid + 1) * B) / NT;
      forward_chunk(P, batch.data() + s, e - s, train_cb[tid]);
    });

    std::vector<double> c_bar(n + 1, 0.0);
    if (proj) {
      std::vector<std::vector<double>> th_cb(NT, std::vector<double>(n + 1, 0.0));
      pool.run_chunks([&](int tid) {
        int s = (tid * Bsym) / NT, e = ((tid + 1) * Bsym) / NT;
        int M = e - s;
        forward_chunk(P, sym_batch.data() + s, M, sym_cb[tid]);
        for (int m = 0; m < M; ++m) {
          for (int t = 1; t <= n; ++t) {
            int pos = n + 1 + t;
            double x_prev = (sym_batch[s + m].states[t - 1] == 0) ? 1.0 : -1.0;
            th_cb[tid][t] += x_prev * (sym_cb[tid].logits[(m * L + pos) * 2 + 0] -
                                       sym_cb[tid].logits[(m * L + pos) * 2 + 1]) / Bsym;
          }
        }
      });
      for (int tid = 0; tid < NT; ++tid) {
        for (int t = 1; t <= n; ++t) c_bar[t] += th_cb[tid][t];
      }
    }

    std::vector<double> train_dlogits(B * L * 2, 0.0);
    std::vector<double> sym_coeff(n + 1, 0.0);
    const double inv_Bn = 1.0 / ((double)B * n);

    for (int tid = 0; tid < NT; ++tid) {
      int s = (tid * B) / NT, e = ((tid + 1) * B) / NT;
      for (int m = 0; m < e - s; ++m) {
        int gm = s + m;
        for (int t = 1; t <= n; ++t) {
          int pos = n + 1 + t;
          double x_prev = (batch[gm].states[t - 1] == 0) ? 1.0 : -1.0;
          double z0 = train_cb[tid].logits[(m * L + pos) * 2 + 0];
          double z1 = train_cb[tid].logits[(m * L + pos) * 2 + 1];
          if (proj) {
            double shift = 0.5 * (c_targ - c_bar[t]) * x_prev;
            z0 += shift;
            z1 -= shift;
          }
          double mx = std::max(z0, z1);
          double e0 = std::exp(z0 - mx), e1 = std::exp(z1 - mx);
          double p0 = e0 / (e0 + e1), p1 = e1 / (e0 + e1);
          int y = batch[gm].states[t];
          double g0 = (p0 - (y == 0 ? 1.0 : 0.0)) * inv_Bn;
          double g1 = (p1 - (y == 1 ? 1.0 : 0.0)) * inv_Bn;
          train_dlogits[(gm * L + pos) * 2 + 0] = g0;
          train_dlogits[(gm * L + pos) * 2 + 1] = g1;
          if (proj) sym_coeff[t] += -0.5 * x_prev * (g0 - g1);
        }
      }
    }

    std::vector<double> sym_dlogits;
    if (proj) {
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

    pool.run_chunks([&](int tid) {
      std::fill(thread_grads[tid].begin(), thread_grads[tid].end(), 0.0);
      int s = (tid * B) / NT, e = ((tid + 1) * B) / NT;
      backward_chunk(P, e - s, train_cb[tid], train_dlogits.data() + s * L * 2, thread_grads[tid].data());
      if (proj) {
        int ss = (tid * Bsym) / NT, ee = ((tid + 1) * Bsym) / NT;
        backward_chunk(P, ee - ss, sym_cb[tid], sym_dlogits.data() + ss * L * 2, thread_grads[tid].data());
      }
    });

    std::fill(grad.begin(), grad.end(), 0.0);
    for (int tid = 0; tid < NT; ++tid) {
      for (size_t i = 0; i < grad.size(); ++i) grad[i] += thread_grads[tid][i];
    }

    double bc1 = 1.0 - std::pow(b1, step + 1);
    double bc2 = 1.0 - std::pow(b2, step + 1);
    for (const auto& b : P.blocks) {
      if (b.name.rfind("B1", 0) == 0 && b.is_lag_bias) continue;
      for (size_t i = 0; i < b.size; ++i) {
        if (b.is_lag_bias && i == 0) continue;
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
}

int main(int argc, char** argv) {
  setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
  Config cfg;
  double p = 0.25;
  int seed = 42;
  int mode_int = 0;
  std::string out_dir = "results";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--p" && i + 1 < argc) p = std::stod(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = std::stoi(argv[++i]);
    else if (a == "--mode" && i + 1 < argc) mode_int = std::stoi(argv[++i]);
    else if (a == "--steps" && i + 1 < argc) cfg.total_steps = std::stoi(argv[++i]);
    else if (a == "--release" && i + 1 < argc) cfg.release_step = std::stoi(argv[++i]);
    else if (a == "--eval-every" && i + 1 < argc) cfg.eval_every = std::stoi(argv[++i]);
    else if (a == "--dmodel" && i + 1 < argc) cfg.d_model = std::stoi(argv[++i]);
    else if (a == "--dk" && i + 1 < argc) cfg.d_k = std::stoi(argv[++i]);
    else if (a == "--dmlp" && i + 1 < argc) cfg.d_mlp = std::stoi(argv[++i]);
    else if (a == "--lr" && i + 1 < argc) cfg.lr = std::stod(argv[++i]);
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
