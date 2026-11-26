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

TARGETS := causal_interventions grad_check step_ode

all: $(TARGETS)

causal_interventions: causal_interventions.cpp cblas_shim.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

grad_check: grad_check.cpp cblas_shim.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

step_ode: step_ode.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -pthread

check: grad_check
	./grad_check --check-only

clean:
	rm -f $(TARGETS)

.PHONY: all check clean
