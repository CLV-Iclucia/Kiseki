// ============================================================================
// include/FluidSim/cpu/cpu-advector.h
// CPUAdvectorP2G / CPUAdvectorG2P: 粒子-网格转换的模块化 Solver
// 内部复用现有 HybridAdvectionSolver3D (PIC/FLIP)
// ============================================================================
#pragma once

#include <FluidSim/cpu/cpu-solver.h>
#include <FluidSim/cpu/advect-solver.h>
#include <memory>

namespace fluid::cpu {

struct AdvectorConfig {
    AdvectionMethod method = AdvectionMethod::FLIP;
    Real flipBlend = 0.97;
    int rk3Steps = 3;
};

class CPUAdvector {
public:
    CPUAdvector(CPUFluidContext& ctx, const AdvectorConfig& config);

    HybridAdvectionSolver3D& impl() { return *advector_; }
    const std::vector<Vec3d>& velocities() const { return advector_->velocities(); }

private:
    std::unique_ptr<HybridAdvectionSolver3D> advector_;
};

// ---- P2G Solver ----
class CPUAdvectorP2G : public CPUModularSolver {
public:
    CPUAdvectorP2G(std::shared_ptr<CPUAdvector> advImpl)
        : CPUModularSolver("cpu_advector_p2g"), impl_(std::move(advImpl)) {}

    void solve(CPUFluidContext& ctx, Real dt) override;

    void configure(const core::Properties& props) override;

private:
    std::shared_ptr<CPUAdvector> impl_;
};

// ---- G2P Solver ----
class CPUAdvectorG2P : public CPUModularSolver {
public:
    CPUAdvectorG2P(std::shared_ptr<CPUAdvector> advImpl)
        : CPUModularSolver("cpu_advector_g2p"), impl_(std::move(advImpl)) {}

    void solve(CPUFluidContext& ctx, Real dt) override;

private:
    std::shared_ptr<CPUAdvector> impl_;
};

} // namespace fluid::cpu
