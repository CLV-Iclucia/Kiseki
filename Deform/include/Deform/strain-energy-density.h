//
// Created by creeper on 5/23/24.
//

#pragma once

#include <Core/profiler.h>
#include <memory>
#include <Deform/types.h>
#include <Deform/invariants.h>
#include <Deform/deformation-gradient.h>
#include <Core/deserializer.h>
#include <unordered_map>
#include <spdlog/spdlog.h>

namespace sim::deform {
template<typename T>
struct StrainEnergyDensity {
  [[nodiscard]] virtual T computeEnergyDensity(const DeformationGradient<T, 3> &dg) const = 0;
  [[nodiscard]] virtual Matrix<T, 3, 3> computeEnergyGradient(const DeformationGradient<T, 3> &dg) const = 0;
  [[nodiscard]] virtual Matrix<T, 9, 9> computeEnergyHessian(const DeformationGradient<T, 3> &dg) const = 0;
  [[nodiscard]] virtual Matrix<T, 9, 9> filteredEnergyHessian(const DeformationGradient<T, 3> &dg) const = 0;
  virtual ~StrainEnergyDensity() = default;
};

template<typename T>
struct ElasticityParameters {
  T E, nu;
};

template<typename T>
struct StableNeoHookean final : StrainEnergyDensity<T> {
  T mu, lambda;
  StableNeoHookean(T mu, T lambda) : mu(mu), lambda(lambda) {}
  explicit StableNeoHookean(ElasticityParameters<T> params) {
    auto [E, nu] = params;
    auto mu_ = E / (2.0 * (1.0 + nu));
    auto lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    mu = 0.4 / 0.3 * mu_;
    lambda = lambda_ + 0.5 / 0.6 * mu_;
  }
  static std::unique_ptr<StrainEnergyDensity<T>> createFromJson(const core::JsonNode &json) {
    if (!json.is<core::JsonDict>())
      throw std::runtime_error("StableNeoHookean requires a JSON object");
    const auto &dict = json.as<core::JsonDict>();
    T mu = dict.at("mu").as<T>();
    T lambda = dict.at("lambda").as<T>();
    return std::make_unique<StableNeoHookean<T>>(mu, lambda);
  }
  T computeEnergyDensity(const DeformationGradient<T, 3> &dg) const override {
    auto I = isotropicInvariants(dg);
    return 0.5 * mu * (I(1) - 3.0) - mu * (I(2) - 1.0) + 0.5 * lambda * (I(2) - 1.0) * (I(2) - 1.0);
  }
  Matrix<T, 3, 3> computeEnergyGradient(const DeformationGradient<T, 3> &dg) const override {
    auto I = isotropicInvariants(dg);
    auto pIiipF = gradientIii<T>(dg);
    auto pIiiipF = gradientIiii<T>(dg);
    return 0.5 * mu * pIiipF - mu * pIiiipF + lambda * (I(2) - 1.0) * pIiiipF;
  }
  Matrix<T, 9, 9> computeEnergyHessian(const DeformationGradient<T, 3> &dg) const override {
    auto I = isotropicInvariants(dg);
    Vector<T, 9> vec_giii = maths::vectorize(gradientIiii<T>(dg));
    return lambda * vec_giii * vec_giii.transpose() + 0.5 * mu * hessianIii<T>()
        + ((I(2) - 1.0) * lambda - mu) * hessianIiii<T>(dg);
  }
  Matrix<T, 9, 9> filteredEnergyHessian(const DeformationGradient<T, 3> &dg) const override {
    SIM_PROFILE_SCOPE("filteredSNHessian");
    auto I = isotropicInvariants(dg);
    Vector<T, 9> vec_giii = maths::vectorize(gradientIiii<T>(dg));
    Matrix<T, 9, 9> H = lambda * vec_giii * vec_giii.transpose() + 0.5 * mu * hessianIii<T>()
        + ((I(2) - 1.0) * lambda - mu) * hessianIiii<T>(dg);
    // compute eigenvalues
    Vector<T, 9> eigenvalues;
    Matrix<T, 9, 9> eigenvectors;
    {
      SIM_PROFILE_SCOPE("SNH_EigenSolve");
      Eigen::SelfAdjointEigenSolver<Matrix<T, 9, 9>> es(H);
      eigenvalues = es.eigenvalues();
      eigenvectors = es.eigenvectors();
    }
    for (int i = 0; i < 9; i++)
      eigenvalues(i) = std::abs(eigenvalues(i));
    return eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
  }
};

template<typename T>
struct ARAP final : StrainEnergyDensity<T> {
  static std::unique_ptr<StrainEnergyDensity<T>> createFromJson(const core::JsonNode &json) {
    return std::make_unique<ARAP<T>>();
  }
  
  T computeEnergyDensity(const DeformationGradient<T, 3> &dg) const override {
    return (dg.F() - dg.R()).squaredNorm();
  }
  Matrix<T, 3, 3> computeEnergyGradient(const DeformationGradient<T, 3> &dg) const override {
    return 2.0 * (dg.F() - dg.R());
  }
  Matrix<T, 9, 9> computeEnergyHessian(const DeformationGradient<T, 3> &dg) const override {
    return hessianIii<T>() - 2 * hessianIi(dg);
  }
  Matrix<T, 9, 9> filteredEnergyHessian(const DeformationGradient<T, 3> &dg) const override {
    SIM_PROFILE_SCOPE("filteredARAPHessian");
    auto H = hessianIii<T>() - 2 * hessianIi(dg);
    Vector<T, 9> eigenvalues;
    Matrix<T, 9, 9> eigenvectors;
    {
      SIM_PROFILE_SCOPE("ARAP_EigenSolve");
      Eigen::SelfAdjointEigenSolver<Matrix<T, 9, 9>> es(H);
      eigenvalues = es.eigenvalues();
      eigenvectors = es.eigenvectors();
    }
    for (int i = 0; i < 9; i++)
      eigenvalues(i) = std::abs(eigenvalues(i));
    return eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
  }
};

template<typename T>
struct LinearElastic final : StrainEnergyDensity<T> {
  T mu, lambda;
  
  LinearElastic(T mu, T lambda) : mu(mu), lambda(lambda) {}
  
  static std::unique_ptr<StrainEnergyDensity<T>> createFromJson(const core::JsonNode &json) {
    if (!json.is<core::JsonDict>())
      throw std::runtime_error("LinearElastic requires a JSON object");
    const auto &dict = json.as<core::JsonDict>();
    T mu = dict.at("mu").as<T>();
    T lambda = dict.at("lambda").as<T>();
    return std::make_unique<LinearElastic<T>>(mu, lambda);
  }
  
  T computeEnergyDensity(const DeformationGradient<T, 3> &dg) const override {
    auto strain = 0.5 * (dg.F() + dg.F().transpose()) - Matrix<T, 3, 3>::Identity();
    T traceStrain = strain.trace();
    return mu * strain.squaredNorm() + 0.5 * lambda * traceStrain * traceStrain;
  }
  
  Matrix<T, 3, 3> computeEnergyGradient(const DeformationGradient<T, 3> &dg) const override {
    auto strain = 0.5 * (dg.F() + dg.F().transpose()) - Matrix<T, 3, 3>::Identity();
    T traceStrain = strain.trace();
    return 2.0 * mu * strain + lambda * traceStrain * Matrix<T, 3, 3>::Identity();
  }
  
  Matrix<T, 9, 9> computeEnergyHessian(const DeformationGradient<T, 3> &dg) const override {
    // 简化的Hessian实现
    return 2.0 * mu * Matrix<T, 9, 9>::Identity() + lambda * Matrix<T, 9, 9>::Ones();
  }
  
  Matrix<T, 9, 9> filteredEnergyHessian(const DeformationGradient<T, 3> &dg) const override {
    return computeEnergyHessian(dg);
  }
};

template<typename T>
std::unique_ptr<StrainEnergyDensity<T>> createStrainEnergyDensity(const core::JsonNode &node);

template std::unique_ptr<StrainEnergyDensity<double>> createStrainEnergyDensity<double>(const core::JsonNode &node);
template std::unique_ptr<StrainEnergyDensity<float>> createStrainEnergyDensity<float>(const core::JsonNode &node);

}



