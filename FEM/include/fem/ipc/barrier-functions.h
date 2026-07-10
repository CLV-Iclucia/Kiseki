//
// Created by creeper on 5/28/24.
//

#ifndef KISEKI_FEM_INCLUDE_FEM_IPC_BARRIER_FUNCTIONS_H_
#define KISEKI_FEM_INCLUDE_FEM_IPC_BARRIER_FUNCTIONS_H_
#include <fem/types.h>
#include <cmath>
namespace ksk::fem::ipc {

struct LogBarrier {
  explicit LogBarrier(Real dHat) : m_dHat(dHat) {}
  [[nodiscard]]
  Real operator()(Real dSqr) const {
    return dSqr >= dHatSqr() ? 0.0 : (dSqr - dHatSqr()) * (dSqr - dHatSqr()) * log(dHatSqr() / dSqr);
  }
  [[nodiscard]] Real distanceSqrGradient(Real dSqr) const {
    return dSqr >= dHatSqr() ? 0.0 : 2 * (dSqr - dHatSqr()) * log(dHatSqr() / dSqr)
        - (dSqr - dHatSqr()) / dSqr * (dSqr - dHatSqr());
  }
  [[nodiscard]] Real distanceSqrHessian(Real dSqr) const {
    return dSqr >= dHatSqr() ? 0.0 : 2 * log(dHatSqr() / dSqr)
        - (dSqr - dHatSqr()) / dSqr * (3 * dSqr + dHatSqr()) / dSqr;
  }
  [[nodiscard]] Real dHat() const { return m_dHat; }
  [[nodiscard]] Real dHatSqr() const { return m_dHat * m_dHat; }

 private:
  Real m_dHat;
};

}
#endif //KISEKI_FEM_INCLUDE_FEM_IPC_BARRIER_FUNCTIONS_H_
