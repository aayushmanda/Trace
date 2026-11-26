UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
  CXX      := clang++
  CXXFLAGS := -O3 -std=c++20 -DACCELERATE_NEW_LAPACK -Wall -Wextra
  LDFLAGS  := -framework Accelerate -pthread
else
  CXX      := g++
  CXXFLAGS := -O3 -std=c++20 -march=native -Wall -Wextra
  LDFLAGS  := -pthread
endif

TARGETS := run_full_plan run_causal_fast run_causal_sweep shortcut_intervention step_ode_suite aggregate_results

all: $(TARGETS)

run_full_plan: run_full_plan.cpp cblas_shim.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run_causal_fast: run_causal_fast.cpp cblas_shim.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run_causal_sweep: run_causal_sweep.cpp cblas_shim.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

shortcut_intervention: shortcut_intervention.cpp cblas_shim.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

step_ode_suite: step_ode_suite.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -pthread

aggregate_results: aggregate_results.cpp
	$(CXX) -O3 -std=c++20 -Wall -Wextra $< -o $@

check: shortcut_intervention
	./shortcut_intervention --check-only

clean:
	rm -f $(TARGETS) ode_extra ode_fixed

.PHONY: all check clean
