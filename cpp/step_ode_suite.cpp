// Exact float64 population solver for (Step-ODE) across:
//   1. Priority 2A: Learned c vs Fixed c == c* (External Offset ODE analog) vs Clamped c == 0
//   2. Priority 3:  Theorem 2(v) threshold sweep p in {0.50, 0.25, 0.15, 0.10, 0.05}
//   3. Priority 6:  Direct numerical audit of the 3 routing sub-lemmas (visible distractor floor,
//                   late distractor floor n >= 5, and Omega_max growth bound) across n in {2,4,6,8,10}.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static inline double sigmoid(double x) {
  if (x >= 0.0) {
    double e = std::exp(-x);
    return 1.0 / (1.0 + e);
  } else {
    double e = std::exp(x);
    return e / (1.0 + e);
  }
}

struct ODEConfig {
  int n = 8;
  int d_k = 24;
  double p = 0.25;
  double eps = 0.08;
  double r_align = 0.5;
  int c_mode = 0; // 0 = learned c(0)=0; 1 = fixed c === c*; 2 = clamped c === 0
};

struct State {
  double c = 0.0;
  double g = 0.0;
  double a = 0.0;
  std::vector<double> beta; // size 2n + 2, beta[0] == 0
};

struct Diagnostics {
  double loss = 0.0;
  double min_pi_star = 1.0;
  double mean_pi_star = 0.0;
  double mean_alpha_lag_n = 0.0; // full-prefix weight on lag n (averaged over heads & steps)
  double mean_alpha_lag_1 = 0.0; // full-prefix weight on lag 1 (averaged over heads & steps)
  double E_dist = 0.0;
  double E_true = 0.0;
  double min_margin = 1e9;
  double dbeta_n = 0.0;
  double dbeta_1 = 0.0;
  double min_vis_n_omega = 1e9;
  double min_late_omega = 1e9;
  double max_omega = 0.0;
};

struct PopPrecomp {
  int n;
  int num_w;
  std::vector<double> prob;      // size 2^n
  std::vector<int> chi;          // size 2^n * n, entries in {-1, +1}
  void init(int n_, double p) {
    n = n_;
    num_w = 1 << n;
    prob.resize(num_w);
    chi.resize(num_w * n);
    for (int mask = 0; mask < num_w; ++mask) {
      double pr = 1.0;
      for (int r = 0; r < n; ++r) {
        int b = (mask >> r) & 1;
        pr *= b ? p : (1.0 - p);
        chi[mask * n + r] = b ? -1 : 1;
      }
      prob[mask] = pr;
    }
  }
};

void compute_rhs_and_diag(
    const ODEConfig& cfg,
    const PopPrecomp& pop,
    double Delta0,
    double v_val,
    const State& st,
    State& dst,
    Diagnostics* diag_out) {

  const int n = cfg.n;
  const int B_sz = 2 * n + 2;
  dst.beta.assign(B_sz, 0.0);

  std::vector<double> exp_b(B_sz);
  exp_b[0] = 1.0;
  for (int d = 1; d < B_sz; ++d) exp_b[d] = std::exp(st.beta[d]);

  double dc_acc = 0.0, dg_acc = 0.0, da_acc = 0.0;
  std::vector<double> dbeta_acc(B_sz, 0.0);

  double loss_sum = 0.0, min_pi = 1.0, sum_pi = 0.0;
  double sum_al_n = 0.0, sum_al_1 = 0.0;
  double ed_sum = 0.0, et_sum = 0.0, min_M = 1e9;

  for (int t = 2; t <= n; ++t) {
    int i_t = n + 1 + t;
    double Z_let = 0.0, Z_other = 0.0;
    std::vector<bool> is_let(i_t + 1, false);
    for (int r = 1; r <= n; ++r) {
      int d = n + t - r;
      is_let[d] = true;
      Z_let += exp_b[d];
    }
    for (int d = 0; d <= i_t; ++d) {
      if (!is_let[d]) Z_other += exp_b[d];
    }
    std::vector<double> pi_t(n + 1, 0.0);
    for (int r = 1; r <= n; ++r) {
      pi_t[r] = exp_b[n + t - r] / Z_let;
    }
    double pi_star = pi_t[t];
    min_pi = std::min(min_pi, pi_star);
    sum_pi += pi_star;

    double u_t = std::log(Z_let / Z_other);
    double sp = sigmoid(u_t + st.a);
    double sm = sigmoid(u_t - st.a);
    double D_t = sp - sm;
    double sp_p = sp * (1.0 - sp);
    double sm_p = sm * (1.0 - sm);
    double xi_t = sp + sm - 1.0;
    double gamma_t = st.g * D_t;

    // Head-averaged full-prefix attention on lag n (letter r=t) and lag 1 (non-letter delta=1)
    double al_let_tot = 0.5 * (sp + sm);
    sum_al_n += al_let_tot * pi_star;
    sum_al_1 += (1.0 - al_let_tot) * (exp_b[1] / Z_other);

    double E_sig_chi = 0.0;
    double mu_t = 0.0;
    std::vector<double> E_sig_chit_chir(n + 1, 0.0);

    for (int mask = 0; mask < pop.num_w; ++mask) {
      const int* ch = &pop.chi[mask * n];
      double rho = 0.0;
      for (int r = 1; r <= n; ++r) rho += pi_t[r] * ch[r - 1];
      int chi_t = ch[t - 1];
      double M_t = chi_t * (st.c + gamma_t * rho);
      if (diag_out) {
        min_M = std::min(min_M, M_t);
        if (M_t > 0) loss_sum += pop.prob[mask] * std::log(1.0 + std::exp(-M_t));
        else loss_sum += pop.prob[mask] * (-M_t + std::log(1.0 + std::exp(M_t)));
      }
      double s_neg = sigmoid(-M_t);
      double w_s = pop.prob[mask] * s_neg;
      E_sig_chi += w_s * chi_t;
      mu_t += w_s * chi_t * rho;
      for (int r = 1; r <= n; ++r) {
        E_sig_chit_chir[r] += w_s * (chi_t * ch[r - 1]);
      }
    }

    dc_acc += E_sig_chi;
    dg_acc += D_t * mu_t;
    da_acc += st.g * (sp_p + sm_p) * mu_t;

    for (int r = 1; r <= n; ++r) {
      int d = n + t - r;
      dbeta_acc[d] += gamma_t * pi_t[r] * (E_sig_chit_chir[r] - (1.0 + xi_t) * mu_t);
      if (diag_out) {
        if (r == t) et_sum += E_sig_chit_chir[r];
        else ed_sum += std::abs(E_sig_chit_chir[r]);
      }
    }
    for (int d = 1; d <= i_t; ++d) {
      if (!is_let[d]) {
        dbeta_acc[d] += gamma_t * (exp_b[d] / Z_other) * xi_t * mu_t;
      }
    }
  }

  double inv_nm1 = 1.0 / (double)(n - 1);
  double eps2 = cfg.eps * cfg.eps;
  double kappa_a = (1.0 / (4.0 * cfg.d_k)) *
                   std::sqrt(eps2 * eps2 * Delta0 * Delta0 + 8.0 * cfg.d_k * st.a * st.a);

  dst.c = (cfg.c_mode == 0) ? (eps2 * dc_acc * inv_nm1) : 0.0;
  dst.g = eps2 * (v_val * v_val) * dg_acc * inv_nm1;
  dst.a = eps2 * kappa_a * da_acc * inv_nm1;
  dst.beta[0] = 0.0;
  for (int d = 1; d < B_sz; ++d) {
    dst.beta[d] = 0.5 * dbeta_acc[d] * inv_nm1;
  }

  if (diag_out) {
    diag_out->loss = loss_sum * inv_nm1;
    diag_out->min_pi_star = min_pi;
    diag_out->mean_pi_star = sum_pi * inv_nm1;
    diag_out->mean_alpha_lag_n = sum_al_n * inv_nm1;
    diag_out->mean_alpha_lag_1 = sum_al_1 * inv_nm1;
    diag_out->E_true = et_sum * inv_nm1;
    diag_out->E_dist = ed_sum / ((double)(n - 1) * (n - 1));
    diag_out->min_margin = min_M;
    diag_out->dbeta_n = dst.beta[n];
    diag_out->dbeta_1 = dst.beta[1];

    double min_vis = 1e9, min_late = 1e9, max_om = 0.0;
    int vis_hi = std::min(n + 3, 2 * n - 1);
    for (int d = 2; d <= vis_hi; ++d) {
      if (d == n) continue;
      double om = std::exp(-st.beta[d]);
      min_vis = std::min(min_vis, n * om);
      max_om = std::max(max_om, om);
    }
    for (int d = n + 4; d <= 2 * n - 1; ++d) {
      double om = std::exp(-st.beta[d]);
      min_late = std::min(min_late, om);
      max_om = std::max(max_om, om);
    }
    diag_out->min_vis_n_omega = min_vis;
    diag_out->min_late_omega = (n >= 5) ? min_late : 1.0;
    diag_out->max_omega = max_om;
  }
}

void add_scaled(const State& a, const State& b, double s, State& out) {
  out.c = a.c + s * b.c;
  out.g = a.g + s * b.g;
  out.a = a.a + s * b.a;
  out.beta.resize(a.beta.size());
  for (size_t i = 0; i < a.beta.size(); ++i) out.beta[i] = a.beta[i] + s * b.beta[i];
}

struct RunSummary {
  double tau_route = -1.0;  // min_t pi_t^* >= 7/8
  double tau_solve = -1.0;  // min_{t,w} M_t >= 0.5
  double final_pi_star = 0.0;
  double final_min_M = 0.0;
  double early_lag_n = 0.0; // when |beta_n| + |beta_1| first hits 0.5
  double early_lag_1 = 0.0;
  double init_ratio_b1_bn = 0.0;
  double inf_vis_n_omega = 1e9;
  double inf_late_omega = 1e9;
  double sup_max_omega = 0.0;
};

RunSummary integrate_ode(const ODEConfig& cfg, const std::string& csv_out = "") {
  PopPrecomp pop;
  pop.init(cfg.n, cfg.p);

  // Set ||q(0)|| = ||k(0)|| = eps, cos_angle = r_align => <q0,k0> = r_align * eps^2
  // Nuclear norm of q0 q0^T - k0 k0^T: singular values are eps^2 * sqrt(1 - r_align^2) repeated twice!
  double qnorm = cfg.eps;
  double v_val = cfg.eps;
  double Delta0 = 2.0 * qnorm * qnorm * std::sqrt(std::max(0.0, 1.0 - cfg.r_align * cfg.r_align));
  double a0 = (cfg.eps * cfg.eps) * (cfg.r_align * qnorm * qnorm) / std::sqrt(2.0 * cfg.d_k);
  double c_star = std::log((1.0 - cfg.p) / cfg.p);

  State st;
  st.c = (cfg.c_mode == 1) ? c_star : 0.0;
  st.g = 0.0;
  st.a = a0;
  st.beta.assign(2 * cfg.n + 2, 0.0);

  std::ofstream ofs;
  if (!csv_out.empty()) {
    ofs.open(csv_out);
    ofs << "tau,loss_bits,c,g,a,pi_star_min,alpha_lag_n,alpha_lag_1,beta_n,beta_1,E_dist,min_M\n";
  }

  RunSummary sum;
  bool captured_init_ratio = false;
  bool captured_early_split = false;
  bool in_routing_window = false;

  double tau = 0.0;
  State k1, k2, k3, k4, tmp;
  Diagnostics diag;

  for (int iter = 0; iter < 180000; ++iter) {
    compute_rhs_and_diag(cfg, pop, Delta0, v_val, st, k1, &diag);

    if (!captured_init_ratio && st.g > 1e-9) {
      sum.init_ratio_b1_bn = diag.dbeta_1 / (std::abs(diag.dbeta_n) + 1e-30);
      captured_init_ratio = true;
    }
    if (cfg.c_mode == 0 && std::abs(sigmoid(-st.c) - cfg.p) <= (cfg.p * (1.0 - cfg.p)) / (256.0 * cfg.n)) {
      in_routing_window = true;
    } else if (cfg.c_mode == 1) {
      in_routing_window = true;
    }
    if (in_routing_window && sum.tau_route < 0.0) {
      sum.inf_vis_n_omega = std::min(sum.inf_vis_n_omega, diag.min_vis_n_omega);
      sum.inf_late_omega = std::min(sum.inf_late_omega, diag.min_late_omega);
      sum.sup_max_omega = std::max(sum.sup_max_omega, diag.max_omega);
    }

    if (!captured_early_split && (std::abs(st.beta[cfg.n]) + std::abs(st.beta[1]) >= 0.6)) {
      sum.early_lag_n = diag.mean_alpha_lag_n;
      sum.early_lag_1 = diag.mean_alpha_lag_1;
      captured_early_split = true;
    }
    if (sum.tau_route < 0.0 && diag.min_pi_star >= 0.875) sum.tau_route = tau;
    if (sum.tau_solve < 0.0 && diag.min_margin >= 0.5) {
      sum.tau_solve = tau;
      if (ofs.is_open()) {
        ofs << tau << "," << diag.loss / std::log(2.0) << "," << st.c << "," << st.g << "," << st.a << ","
            << diag.min_pi_star << "," << diag.mean_alpha_lag_n << "," << diag.mean_alpha_lag_1 << ","
            << st.beta[cfg.n] << "," << st.beta[1] << "," << diag.E_dist << "," << diag.min_margin << "\n";
      }
      sum.final_pi_star = diag.min_pi_star;
      sum.final_min_M = diag.min_margin;
      break;
    }

    if (ofs.is_open() && (iter % 150 == 0)) {
      ofs << tau << "," << diag.loss / std::log(2.0) << "," << st.c << "," << st.g << "," << st.a << ","
          << diag.min_pi_star << "," << diag.mean_alpha_lag_n << "," << diag.mean_alpha_lag_1 << ","
          << st.beta[cfg.n] << "," << st.beta[1] << "," << diag.E_dist << "," << diag.min_margin << "\n";
    }

    // Adaptive step size limiting max coordinate change per RK4 step
    double max_rate = std::max({std::abs(k1.c) / 0.05,
                                std::abs(k1.g) / 0.03,
                                std::abs(k1.a) / std::max(0.02 * std::abs(st.a), 1e-10)});
    for (int d = 1; d < 2 * cfg.n + 2; ++d) {
      max_rate = std::max(max_rate, std::abs(k1.beta[d]) / 0.04);
    }
    double dt = (max_rate > 1e-30) ? (1.0 / max_rate) : 1.0;
    dt = std::min(dt, 5e6);

    add_scaled(st, k1, 0.5 * dt, tmp);
    compute_rhs_and_diag(cfg, pop, Delta0, v_val, tmp, k2, nullptr);
    add_scaled(st, k2, 0.5 * dt, tmp);
    compute_rhs_and_diag(cfg, pop, Delta0, v_val, tmp, k3, nullptr);
    add_scaled(st, k3, dt, tmp);
    compute_rhs_and_diag(cfg, pop, Delta0, v_val, tmp, k4, nullptr);

    st.c += (dt / 6.0) * (k1.c + 2 * k2.c + 2 * k3.c + k4.c);
    st.g += (dt / 6.0) * (k1.g + 2 * k2.g + 2 * k3.g + k4.g);
    st.a += (dt / 6.0) * (k1.a + 2 * k2.a + 2 * k3.a + k4.a);
    for (size_t d = 1; d < st.beta.size(); ++d) {
      st.beta[d] += (dt / 6.0) * (k1.beta[d] + 2 * k2.beta[d] + 2 * k3.beta[d] + k4.beta[d]);
    }
    tau += dt;
    sum.final_pi_star = diag.min_pi_star;
    sum.final_min_M = diag.min_margin;
  }
  return sum;
}

int main() {
  std::cout << "=== 1. Priority 2A: Step-ODE External-Offset Control (n=8, dk=24, p=0.25, eps=0.08) ===\n";
  const char* mode_names[3] = {"Learned c(0)=0", "Fixed c === c*", "Clamped c === 0"};
  for (int m = 0; m < 3; ++m) {
    ODEConfig cfg{8, 24, 0.25, 0.08, 0.5, m};
    std::string fn = "results_ode_p025_m" + std::to_string(m) + ".csv";
    RunSummary s = integrate_ode(cfg, fn);
    std::cout << std::left << std::setw(18) << mode_names[m]
              << " tau_route=" << std::scientific << std::setprecision(3) << s.tau_route
              << " tau_solve=" << s.tau_solve
              << " final_pi*=" << std::fixed << std::setprecision(4) << s.final_pi_star
              << " final_M=" << s.final_min_M << "\n";
  }

  std::cout << "\n=== 2. Priority 3: Theorem 2(v) Threshold Sweep (n=8, dk=24, eps=0.08) ===\n";
  double ps[5] = {0.50, 0.25, 0.15, 0.10, 0.05};
  for (double p : ps) {
    double lam2 = (1.0 - 2.0 * p) * (1.0 - 2.0 * p);
    ODEConfig c_learn{8, 24, p, 0.08, 0.5, 0};
    ODEConfig c_clamp{8, 24, p, 0.08, 0.5, 2};
    std::string tag = std::to_string((int)std::round(p * 100));
    RunSummary sl = integrate_ode(c_learn, "results_ode_learn_p" + tag + ".csv");
    RunSummary sc = integrate_ode(c_clamp, "results_ode_clamp_p" + tag + ".csv");
    std::cout << "p=" << std::fixed << std::setprecision(2) << p << " (lam^2=" << lam2 << ")"
              << " | Clamped init dbeta1/dbetan=" << std::setprecision(3) << sc.init_ratio_b1_bn
              << " early(lag_n, lag_1)=(" << sc.early_lag_n << ", " << sc.early_lag_1 << ")"
              << " | Solve ratio (clamp/learn)="
              << ((sc.tau_solve > 0 && sl.tau_solve > 0) ? (sc.tau_solve / sl.tau_solve) : -1.0)
              << " (clamp_solve=" << std::scientific << sc.tau_solve
              << ", learn_solve=" << sl.tau_solve << ")\n";
  }

  std::cout << "\n=== 3. Priority 6: Numerical Audit of Routing Sub-Lemma Constants ===\n";
  int ns[5] = {2, 4, 6, 8, 10};
  double p_aud[3] = {0.10, 0.25, 0.50};
  double global_min_vis = 1e9, global_min_late = 1e9, global_max_om = 0.0;
  for (int n : ns) {
    for (double p : p_aud) {
      ODEConfig cfg{n, 2 * n + 6, p, 0.08, 0.5, 0};
      RunSummary s = integrate_ode(cfg);
      global_min_vis = std::min(global_min_vis, s.inf_vis_n_omega);
      if (n >= 5) global_min_late = std::min(global_min_late, s.inf_late_omega);
      global_max_om = std::max(global_max_om, s.sup_max_omega);
      std::cout << "n=" << std::setw(2) << n << " p=" << std::fixed << std::setprecision(2) << p
                << " | inf(n*omega_vis)=" << std::setprecision(4) << s.inf_vis_n_omega
                << " inf(omega_late)=" << s.inf_late_omega
                << " sup(Omega_max)=" << s.sup_max_omega << "\n";
    }
  }
  std::cout << "SUMMARY BOUNDS: inf(n*omega_vis)=" << global_min_vis
            << ", inf_{n>=5}(omega_late)=" << global_min_late
            << ", sup(Omega_max)=" << global_max_om << "\n";
  return 0;
}
