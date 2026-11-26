// Pure C++20 aggregator for causal shortcut-intervention CSV trajectories.
// Computes multi-seed mean, min, max, and solve-time statistics and prints summary tables + PGFPlots coordinates.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct Row {
  int step = 0;
  double loss = 0, loss_bits = 0, one_step_acc = 0, rollout_acc = 0;
  double c_sym = 0, lag_n_share = 0, E_dist = 0, E_true = 0;
};

std::vector<Row> read_csv(const std::string& path) {
  std::vector<Row> rows;
  std::ifstream ifs(path);
  if (!ifs.is_open()) return rows;
  std::string line;
  std::getline(ifs, line); // header
  while (std::getline(ifs, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string item;
    std::vector<double> vals;
    while (std::getline(ss, item, ',')) vals.push_back(std::stod(item));
    if (vals.size() >= 9) {
      Row r;
      r.step = (int)std::llround(vals[0]);
      r.loss = vals[1];
      r.loss_bits = vals[2];
      r.one_step_acc = vals[3];
      r.rollout_acc = vals[4];
      r.c_sym = vals[5];
      r.lag_n_share = vals[6];
      r.E_dist = vals[7];
      r.E_true = vals[8];
      rows.push_back(r);
    }
  }
  return rows;
}

int main(int argc, char** argv) {
  std::string dir = (argc > 1) ? argv[1] : "results";
  std::vector<std::string> p_tags = {"0.25", "0.50", "0.75"};
  std::vector<int> modes = {0, 1, 2, 3, 4};
  std::vector<int> seeds = {42, 43, 44, 45, 46};
  const char* mode_names[5] = {
    "Natural",
    "Suppressed (c=0)",
    "Ext-Calibrated (+c*)",
    "Restored",
    "Weight-Calibrated V(0)=c*"
  };

  for (const auto& p : p_tags) {
    bool any_p = false;
    for (int m : modes) {
      std::vector<std::vector<Row>> seed_runs;
      for (int s : seeds) {
        std::string fn = dir + "/p" + p + "_m" + std::to_string(m) + "_s" + std::to_string(s) + ".csv";
        auto r = read_csv(fn);
        if (!r.empty()) seed_runs.push_back(r);
      }
      if (seed_runs.empty()) continue;
      if (!any_p) {
        std::cout << "\n==================== p = " << p << " ====================\n";
        any_p = true;
      }
      int S = (int)seed_runs.size();
      int T = (int)seed_runs[0].size();
      for (int s = 1; s < S; ++s) T = std::min(T, (int)seed_runs[s].size());
      int solved_cnt = 0;
      std::vector<int> solve_steps;
      for (int s = 0; s < S; ++s) {
        int sol = -1;
        for (const auto& rw : seed_runs[s]) {
          if (rw.rollout_acc >= 99.0) {
            sol = rw.step;
            break;
          }
        }
        if (sol >= 0) {
          solved_cnt++;
          solve_steps.push_back(sol);
        }
      }
      std::sort(solve_steps.begin(), solve_steps.end());
      std::cout << "\n--- Mode " << m << " [" << mode_names[m] << "] (" << S << " seeds, solved >=99%: "
                << solved_cnt << "/" << S << ") ---\n";
      std::cout << "step | loss_bits (min..max) | roll% (min..max) | lag_n (min..max) | Edist  | Etrue  | c_sym\n";
      for (int t = 0; t < T; ++t) {
        double lb = 0, lb_lo = 1e9, lb_hi = -1e9;
        double rl = 0, rl_lo = 1e9, rl_hi = -1e9;
        double lg = 0, lg_lo = 1e9, lg_hi = -1e9;
        double ed = 0, et = 0, cs = 0;
        for (int s = 0; s < S; ++s) {
          const auto& rw = seed_runs[s][t];
          lb += rw.loss_bits / S;
          lb_lo = std::min(lb_lo, rw.loss_bits);
          lb_hi = std::max(lb_hi, rw.loss_bits);
          rl += rw.rollout_acc / S;
          rl_lo = std::min(rl_lo, rw.rollout_acc);
          rl_hi = std::max(rl_hi, rw.rollout_acc);
          lg += rw.lag_n_share / S;
          lg_lo = std::min(lg_lo, rw.lag_n_share);
          lg_hi = std::max(lg_hi, rw.lag_n_share);
          ed += rw.E_dist / S;
          et += rw.E_true / S;
          cs += rw.c_sym / S;
        }
        std::cout << std::setw(4) << seed_runs[0][t].step << " | "
                  << std::fixed << std::setprecision(3) << lb << " (" << lb_lo << ".." << lb_hi << ") | "
                  << std::setprecision(1) << std::setw(5) << rl << " (" << rl_lo << ".." << rl_hi << ") | "
                  << std::setprecision(3) << lg << " (" << lg_lo << ".." << lg_hi << ") | "
                  << std::setprecision(4) << ed << " | " << et << " | " << std::setprecision(3) << cs << "\n";
      }
    }
  }
  return 0;
}
