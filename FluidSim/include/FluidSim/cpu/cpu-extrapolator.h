// ============================================================================
// include/FluidSim/cpu/cpu-extrapolator.h
// CPUExtrapolator: 速度外推模块化 Solver
// ============================================================================
#pragma once

#include <FluidSim/cpu/cpu-solver.h>

namespace fluid::cpu {

struct ExtrapolatorConfig {
    int iters = 10;
};

class CPUExtrapolator : public CPUModularSolver {
public:
    explicit CPUExtrapolator(CPUFluidContext& ctx,
                             ExtrapolatorConfig config = {})
        : CPUModularSolver("cpu_extrapolator"), config_(config)
    {
        (void)ctx;
    }

    void solve(CPUFluidContext& ctx, Real dt) override;

    void configure(const core::Properties& props) override;

private:
    ExtrapolatorConfig config_;

    template<typename GridType>
    void extrapolate(std::unique_ptr<GridType>& g,
                     std::unique_ptr<GridType>& gbuf,
                     std::unique_ptr<Array3D<char>>& valid,
                     std::unique_ptr<Array3D<char>>& validBuf, int iters);
};

// ===== 模板方法实现 =====
template<typename GridType>
void CPUExtrapolator::extrapolate(
    std::unique_ptr<GridType>& g,
    std::unique_ptr<GridType>& gbuf,
    std::unique_ptr<Array3D<char>>& valid,
    std::unique_ptr<Array3D<char>>& validBuf, int iters)
{
    for (int iter = 0; iter < iters; iter++) {
        validBuf->fill(false);
        g->parallelForEach([&](int i, int j, int k) {
            if (valid->at(i, j, k)) {
                validBuf->at(i, j, k) = true;
                return;
            }
            Real sum{0.0};
            int count{0};
            if (i > 0 && valid->at(i - 1, j, k)) {
                sum += g->at(i - 1, j, k);
                count++;
            }
            if (i < g->width() - 1 && valid->at(i + 1, j, k)) {
                sum += g->at(i + 1, j, k);
                count++;
            }
            if (j > 0 && valid->at(i, j - 1, k)) {
                sum += g->at(i, j - 1, k);
                count++;
            }
            if (j < g->height() - 1 && valid->at(i, j + 1, k)) {
                sum += g->at(i, j + 1, k);
                count++;
            }
            if (k > 0 && valid->at(i, j, k - 1)) {
                sum += g->at(i, j, k - 1);
                count++;
            }
            if (k < g->depth() - 1 && valid->at(i, j, k + 1)) {
                sum += g->at(i, j, k + 1);
                count++;
            }
            if (count > 0) {
                gbuf->at(i, j, k) = sum / count;
                validBuf->at(i, j, k) = true;
            } else {
                gbuf->at(i, j, k) = 0.0;
                validBuf->at(i, j, k) = false;
            }
        });
        std::swap(g, gbuf);
        std::swap(valid, validBuf);
    }
}

} // namespace fluid::cpu
