#include "kernels/residual_ls.h"
#include <cstdio>
#include <random>
#include <vector>

using namespace winner::kernels;

int main() {
    const int dim = 256, rank = 64, probes = 32;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<float> W(dim * dim);
    for (auto& v : W) v = nd(rng);

    auto F0 = pack_ternary(W.data(), dim, dim, -1.f);
    auto R = fit_residual_ls(W.data(), F0, rank, 10, 42);

    printf("F0 threshold=%.2f  residual rank=%d\n", F0.threshold, R.rank);
    printf("top singular: ");
    for (int i = 0; i < std::min(5, R.rank); ++i) printf("%.3f ", R.singular[i]);
    printf("\n");

    double cos_f0 = 0, cos_ls = 0, nrmse_f0 = 0, nrmse_ls = 0;
    std::vector<float> x(dim), y_true(dim), y_f0(dim), y_ls(dim);
    for (int p = 0; p < probes; ++p) {
        for (auto& v : x) v = nd(rng);
        // dense ref
        for (int r = 0; r < dim; ++r) {
            float s = 0; for (int c = 0; c < dim; ++c) s += W[r*dim+c]*x[c];
            y_true[r] = s;
        }
        gemv_ternary(F0, x.data(), y_f0.data());
        gemv_f0_plus_residual(F0, R, x.data(), y_ls.data());
        cos_f0 += cosine_similarity(y_true.data(), y_f0.data(), dim);
        cos_ls += cosine_similarity(y_true.data(), y_ls.data(), dim);
        nrmse_f0 += nrmse_metric(y_true.data(), y_f0.data(), dim);
        nrmse_ls += nrmse_metric(y_true.data(), y_ls.data(), dim);
    }
    printf("F0 only:     cos=%.6f  nrmse=%.6f\n", cos_f0/probes, nrmse_f0/probes);
    printf("F0 + LS r%d: cos=%.6f  nrmse=%.6f\n", rank, cos_ls/probes, nrmse_ls/probes);
    printf("OK module residual_ls\n");
    return 0;
}
