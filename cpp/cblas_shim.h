// Portable replacement for the three cblas_sgemm layouts used by the experiment drivers
// (row-major NN, NT, TN). Lets the Apple-Accelerate sources build on Linux:
//   g++ -O3 -march=native -std=c++20 -DUSE_CBLAS_SHIM -include cblas_shim.h file.cpp -pthread
#pragma once
#include <cstring>
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };

// C (MxN) = alpha * op(A) op(B) + beta * C, row-major, lda/ldb/ldc as in CBLAS.
inline void cblas_sgemm(CBLAS_ORDER, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int M, int N, int K,
                        float alpha, const float* A, int lda, const float* B, int ldb, float beta,
                        float* C, int ldc) {
  for (int i = 0; i < M; ++i) {
    float* Ci = C + (long)i * ldc;
    if (beta == 0.0f) std::memset(Ci, 0, sizeof(float) * N);
    else if (beta != 1.0f) for (int j = 0; j < N; ++j) Ci[j] *= beta;
  }
  if (ta == CblasNoTrans && tb == CblasNoTrans) {        // A: MxK, B: KxN
    for (int i = 0; i < M; ++i) {
      const float* Ai = A + (long)i * lda;
      float* Ci = C + (long)i * ldc;
      for (int k = 0; k < K; ++k) {
        const float a = alpha * Ai[k];
        if (a == 0.0f) continue;
        const float* Bk = B + (long)k * ldb;
        for (int j = 0; j < N; ++j) Ci[j] += a * Bk[j];
      }
    }
  } else if (ta == CblasNoTrans && tb == CblasTrans) {   // A: MxK, B: NxK
    for (int i = 0; i < M; ++i) {
      const float* Ai = A + (long)i * lda;
      float* Ci = C + (long)i * ldc;
      for (int j = 0; j < N; ++j) {
        const float* Bj = B + (long)j * ldb;
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s += Ai[k] * Bj[k];
        Ci[j] += alpha * s;
      }
    }
  } else if (ta == CblasTrans && tb == CblasNoTrans) {   // A: KxM, B: KxN
    for (int k = 0; k < K; ++k) {
      const float* Ak = A + (long)k * lda;
      const float* Bk = B + (long)k * ldb;
      for (int i = 0; i < M; ++i) {
        const float a = alpha * Ak[i];
        if (a == 0.0f) continue;
        float* Ci = C + (long)i * ldc;
        for (int j = 0; j < N; ++j) Ci[j] += a * Bk[j];
      }
    }
  } else {                                               // A: KxM, B: NxK
    for (int i = 0; i < M; ++i) {
      float* Ci = C + (long)i * ldc;
      for (int j = 0; j < N; ++j) {
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s += A[(long)k * lda + i] * B[(long)j * ldb + k];
        Ci[j] += alpha * s;
      }
    }
  }
}
inline void cblas_dgemm(CBLAS_ORDER, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int M, int N, int K,
                        double alpha, const double* A, int lda, const double* B, int ldb, double beta,
                        double* C, int ldc) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double s = 0.0;
      for (int k = 0; k < K; ++k) {
        double a = (ta == CblasNoTrans) ? A[(long)i * lda + k] : A[(long)k * lda + i];
        double b = (tb == CblasNoTrans) ? B[(long)k * ldb + j] : B[(long)j * ldb + k];
        s += a * b;
      }
      C[(long)i * ldc + j] = alpha * s + (beta == 0.0 ? 0.0 : beta * C[(long)i * ldc + j]);
    }
}
