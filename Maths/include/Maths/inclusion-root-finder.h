//
// Created by creeper on 9/10/24.
//

#pragma once
#include <optional>
#include <Maths/types.h>
#include <queue>
namespace ksk::maths {

struct Interval {
  std::array<Real, 2> bounds{};
};

inline Real intervalWidth(const Interval &I) {
  return I.bounds[1] - I.bounds[0];
}

template<size_t Dim>
using MultiInterval = std::array<Interval, Dim>;

template<size_t Dim>
inline Real intervalWidth(const MultiInterval<Dim> &I) {
  Real w = 0.0;
  for (int i = 0; i < Dim; i++)
    w = std::max(intervalWidth(I[i]));
  return w;
}

template<typename Func>
std::pair<MultiInterval<3>, MultiInterval<3>> split(Func &&F, const MultiInterval<3> &I) {
  std::array<Interval, 3> I1{};
  std::array<Interval, 3> I2{};
  for (int i = 0; i < 3; i++) {
    Real mid = 0.5 * (I[i].bounds[0] + I[i].bounds[1]);
    I1[i].bounds[0] = I[i].bounds[0];
    I1[i].bounds[1] = mid;
    I2[i].bounds[0] = mid;
    I2[i].bounds[1] = I[i].bounds[1];
  }
  return {I1, I2};
}

template<typename Func>
struct BoxInclusionFunction {
  Func &&func;
  MultiInterval<3> operator()(const MultiInterval<3> &I) const {
    std::array<Real, 8> vx{};
    std::array<Real, 8> vy{};
    std::array<Real, 8> vz{};
    for (int i = 0; i < 8; i++) {
      int m = i & 1;
      int n = (i >> 1) & 1;
      int l = (i >> 2) & 1;
      Real tm = I[0].bounds[m];
      Real an = I[1].bounds[n];
      Real bl = I[2].bounds[l];
      auto v = func(tm, an, bl);
      vx[i] = v(0);
      vy[i] = v(1);
      vz[i] = v(2);
    }
    Real mx = *std::min_element(vx.begin(), vx.end());
    Real Mx = *std::max_element(vx.begin(), vx.end());
    Real my = *std::min_element(vy.begin(), vy.end());
    Real My = *std::max_element(vy.begin(), vy.end());
    Real mz = *std::min_element(vz.begin(), vz.end());
    Real Mz = *std::max_element(vz.begin(), vz.end());
    return {Interval{.bounds = {mx, Mx}},
            Interval{.bounds = {my, My}},
            Interval{.bounds = {mz, Mz}}};
  }
};

struct CandidateInterval {
  MultiInterval<3> I{};
  int level{};
};

struct IntervalOrder {
  bool operator()(const CandidateInterval &a, const CandidateInterval &b) const {
    if (a.level == b.level)
      return a.I[0].bounds[0] < b.I[0].bounds[0];
    return a.level < b.level;
  }
};

struct TightInclusionSolverConfig {
  Real delta{1e-6};
  Real toi{1.0};
  std::optional<int> maxIterations{};
  mutable std::priority_queue<CandidateInterval, std::vector<CandidateInterval>, IntervalOrder> intervals;
};

inline bool intervalIntersect(const MultiInterval<3> &I1, const MultiInterval<3> &I2) {
  for (int i = 0; i < 3; i++) {
    if (I1[i].bounds[1] < I2[i].bounds[0] || I1[i].bounds[0] > I2[i].bounds[1])
      return false;
  }
  return true;
}

inline bool insideInterval(const MultiInterval<3> &I1, const MultiInterval<3> &I2) {
  for (int i = 0; i < 3; i++) {
    if (I1[i].bounds[0] < I2[i].bounds[0] || I1[i].bounds[1] > I2[i].bounds[1])
      return false;
  }
  return true;
}

template<typename Func>
std::optional<Real> tightInclusionSolve(Func &&F, const TightInclusionSolverConfig &config) {
  int nIter = 0;
  auto &intervals = config.intervals;
  while (!intervals.empty()) intervals.pop();
  auto I0 = MultiInterval<3>{Interval{.bounds = {0.0, config.toi}}, Interval{.bounds = {0.0, 1.0}},
                             Interval{.bounds = {0.0, 1.0}}};
  intervals.push({I0, 0});
  int lp = -1;
  auto BF = BoxInclusionFunction{F};
  auto Ce = F.cubeEpsilon(I0);
  Interval If{};
  while (!intervals.empty()) {
    auto [I, l] = intervals.top();
    intervals.pop();
    auto B = BF(I);
    nIter++;
    if (!intervalIntersect(B, Ce)) continue;
    if (l != lp) If = I[0];
    if (nIter >= config.maxIterations)
      return If.bounds[0];
    if (intervalWidth(B) < config.delta || insideInterval(B, Ce)) {
      if (l != lp)
        return If.bounds[0];
    } else {
      auto [I1, I2] = split(F, I);
      intervals.push({I1, l + 1});
      intervals.push({I2, l + 1});
    }
    lp = l;
  }
  return std::nullopt;
}
}
