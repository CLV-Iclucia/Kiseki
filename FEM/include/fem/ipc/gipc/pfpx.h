//
// GIPC PFPx 构造器 — 为每种接触对构造人造形变梯度 F 和 ∂vec(F)/∂x
//
// 核心思想（GIPC #else 路径）:
//   1. 把法向投影到固定方向，构造人造 Dm（使距离=d̂ 时 I₅=1）
//   2. F = Ds · Dm⁻¹
//   3. I₅ = ||F·n||² = (d/d̂)²
//   4. PFPx = ∂vec(F)/∂x 是线性常数矩阵
//
// 输出可直接用于 GIPC 的 gradient sandwich 和 Hessian sandwich
//

#pragma once
#include <fem/types.h>
#include <Eigen/Dense>
#include <glm/glm.hpp>
#include <glm/geometric.hpp>

namespace ksk::fem::ipc::gipc {

// ============================================================================
// 结果结构：存储 PFPx + 归一化特征向量 q₀ + I₅
// ============================================================================

/// 4 顶点 (EE/PT): 12 DOF, vec(F) 9 维
struct PFPxResult12 {
  Eigen::Matrix<Real, 9, 12> PFPx;   // ∂vec(F)/∂x
  Eigen::Matrix<Real, 9, 1> q0;      // 归一化特征向量 = vec(F·n·nᵀ) / √I₅
  Real I5;                            // = (d/d̂)²
  bool valid = false;                 // 构造是否成功
};

/// 3 顶点 (PE): 9 DOF, vec(F) 6 维
struct PFPxResult9 {
  Eigen::Matrix<Real, 6, 9> PFPx;
  Eigen::Matrix<Real, 6, 1> q0;
  Real I5;
  bool valid = false;
};

/// 2 顶点 (PP): 6 DOF, vec(F) 3 维
struct PFPxResult6 {
  Eigen::Matrix<Real, 3, 6> PFPx;
  Eigen::Matrix<Real, 3, 1> q0;
  Real I5;
  bool valid = false;
};

// ============================================================================
// PFPx 构造函数
// ============================================================================

/// Edge-Edge: x = [ea0; ea1; eb0; eb1], 12 DOF
/// Ds = [ea1-ea0 | eb0-ea0 | eb1-ea0], 人造 Dm 使法向距离 = d̂
PFPxResult12 computePFPx_EE(
    const glm::dvec3& ea0, const glm::dvec3& ea1,
    const glm::dvec3& eb0, const glm::dvec3& eb1,
    Real dHat);

/// Edge-Edge mollifier 分支 (PEE): x = [ea0; ea1; eb0; eb1], 12 DOF
/// 使用 GIPC 的 pFpx_pee autogen 代码，适用于近平行边对
/// 输出 PFPx (9×12) 但不计算 q0/I5（mollifier 分支不用 q0）
PFPxResult12 computePFPx_PEE(
    const glm::dvec3& ea0, const glm::dvec3& ea1,
    const glm::dvec3& eb0, const glm::dvec3& eb1,
    Real dHat);

/// Point-Triangle: x = [p; t0; t1; t2], 12 DOF
/// Ds = [t0-p | t1-p | t2-p], 人造 Dm 使法向距离 = d̂
PFPxResult12 computePFPx_PT(
    const glm::dvec3& p,
    const glm::dvec3& t0, const glm::dvec3& t1, const glm::dvec3& t2,
    Real dHat);

/// Point-Edge: x = [p; e0; e1], 9 DOF
/// Ds(3×2) = [e0-p | e1-p], F(3×2), vec(F) 6维
PFPxResult9 computePFPx_PE(
    const glm::dvec3& p,
    const glm::dvec3& e0, const glm::dvec3& e1,
    Real dHat);

/// Point-Point: x = [p0; p1], 6 DOF
/// Ds(3×1) = p1-p0, F(3×1), vec(F) 3维
PFPxResult6 computePFPx_PP(
    const glm::dvec3& p0, const glm::dvec3& p1,
    Real dHat);

// ============================================================================
// 内部辅助：通用 PFPx 矩阵计算 (与 Deform/deformation-gradient.h 等价逻辑)
// ============================================================================
namespace detail {

/// 3D: F = Ds(3×3) · DmInv(3×3), x = [x0;x1;x2;x3]
/// Ds = [x1-x0 | x2-x0 | x3-x0]
/// 输出 9×12 矩阵 ∂vec(F)/∂x
Eigen::Matrix<Real, 9, 12> computePFPx3D(const Eigen::Matrix3d& DmInv);

/// 2D: F = Ds(3×2) · DmInv(2×2), x = [x0;x1;x2]
/// Ds = [x1-x0 | x2-x0] (3×2)
/// 输出 6×9 矩阵
Eigen::Matrix<Real, 6, 9> computePFPx3x2(const Eigen::Matrix2d& DmInv);

/// 1D: F = Ds(3×1) · DmInv(标量), x = [x0;x1]
/// Ds = x1-x0 (3×1)
/// 输出 3×6 矩阵
Eigen::Matrix<Real, 3, 6> computePFPx3x1(Real DmInv);

} // namespace detail

} // namespace ksk::fem::ipc::gipc
