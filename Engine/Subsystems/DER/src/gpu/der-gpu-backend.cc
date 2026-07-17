#include <DER/gpu/der-gpu-backend.h>

#include <DER/der-subsystem.h>

#include <stdexcept>
#include <string>

namespace ksk::der {

DERGpuBackend::DERGpuBackend(DERSubsystem& subsystem)
    : subsystem_(subsystem)
{
}

void DERGpuBackend::writeState(runtime::DofView q,
                               runtime::DofView qdot) const
{
  requireGPU(q, "DERGpuBackend::writeState");
  requireGPU(qdot, "DERGpuBackend::writeState");
  requireSameDevice(q, qdot, "DERGpuBackend::writeState");
  unsupported("DERGpuBackend::writeState");
}

void DERGpuBackend::readState(runtime::ConstDofView q,
                              runtime::ConstDofView qdot)
{
  requireGPU(q, "DERGpuBackend::readState");
  requireGPU(qdot, "DERGpuBackend::readState");
  requireSameDevice(q, qdot, "DERGpuBackend::readState");
  unsupported("DERGpuBackend::readState");
}

void DERGpuBackend::beginStep(runtime::ConstDofView q,
                              runtime::ConstDofView qdot,
                              double dt)
{
  (void)dt;
  requireGPU(q, "DERGpuBackend::beginStep");
  requireGPU(qdot, "DERGpuBackend::beginStep");
  requireSameDevice(q, qdot, "DERGpuBackend::beginStep");
  unsupported("DERGpuBackend::beginStep");
}

void DERGpuBackend::acceptStep(runtime::ConstDofView q,
                               runtime::DofView qdot,
                               double dt)
{
  (void)dt;
  requireGPU(q, "DERGpuBackend::acceptStep");
  requireGPU(qdot, "DERGpuBackend::acceptStep");
  requireSameDevice(q, qdot, "DERGpuBackend::acceptStep");
  unsupported("DERGpuBackend::acceptStep");
}

double DERGpuBackend::evaluateObjective(runtime::ConstDofView q,
                                        runtime::ConstDofView qdot,
                                        double dt)
{
  (void)dt;
  requireGPU(q, "DERGpuBackend::evaluateObjective");
  requireGPU(qdot, "DERGpuBackend::evaluateObjective");
  requireSameDevice(q, qdot, "DERGpuBackend::evaluateObjective");
  unsupported("DERGpuBackend::evaluateObjective");
}

void DERGpuBackend::updateInternalConstraints(double time, double dt)
{
  (void)time;
  (void)dt;
}

void DERGpuBackend::prepareLocalOperator(double dt)
{
  (void)dt;
  unsupported("DERGpuBackend::prepareLocalOperator");
}

void DERGpuBackend::assembleLocalGradient(runtime::DofView g) const
{
  requireGPU(g, "DERGpuBackend::assembleLocalGradient");
  unsupported("DERGpuBackend::assembleLocalGradient");
}

void DERGpuBackend::applyLocalMatrix(runtime::ConstDofView x,
                                     runtime::DofView y) const
{
  requireGPU(x, "DERGpuBackend::applyLocalMatrix");
  requireGPU(y, "DERGpuBackend::applyLocalMatrix");
  requireSameDevice(x, y, "DERGpuBackend::applyLocalMatrix");
  unsupported("DERGpuBackend::applyLocalMatrix");
}

void DERGpuBackend::solveLocalSystem(runtime::ConstDofView b,
                                     runtime::DofView x) const
{
  requireGPU(b, "DERGpuBackend::solveLocalSystem");
  requireGPU(x, "DERGpuBackend::solveLocalSystem");
  requireSameDevice(b, x, "DERGpuBackend::solveLocalSystem");
  unsupported("DERGpuBackend::solveLocalSystem");
}

void DERGpuBackend::mapLocalDirectionToGeometry(runtime::ConstDofView localDq,
                                           runtime::GeometryView globalDx) const
{
  requireGPU(localDq, "DERGpuBackend::mapLocalDirectionToGeometry");
  requireGPU(globalDx, "DERGpuBackend::mapLocalDirectionToGeometry");
  requireSameDevice(localDq, globalDx, "DERGpuBackend::mapLocalDirectionToGeometry");
  unsupported("DERGpuBackend::mapLocalDirectionToGeometry");
}

void DERGpuBackend::scatterContactGradient(
    std::span<const runtime::PointIdx> points,
    runtime::ConstGeometryView pointGradient,
    runtime::DofView g) const
{
  (void)points;
  requireGPU(pointGradient, "DERGpuBackend::scatterContactGradient");
  requireGPU(g, "DERGpuBackend::scatterContactGradient");
  if (pointGradient.device() != g.device()) {
    throw std::runtime_error(
        "DERGpuBackend::scatterContactGradient received buffers on "
        "different devices");
  }
  unsupported("DERGpuBackend::scatterContactGradient");
}

void DERGpuBackend::applyInternalContactHessian(
    runtime::ConstDofView localDq,
    runtime::DofView localY) const
{
  requireGPU(localDq, "DERGpuBackend::applyInternalContactHessian");
  requireGPU(localY, "DERGpuBackend::applyInternalContactHessian");
  requireSameDevice(localDq, localY,
                    "DERGpuBackend::applyInternalContactHessian");
  unsupported("DERGpuBackend::applyInternalContactHessian");
}

void DERGpuBackend::unsupported(const char* operation)
{
  throw std::runtime_error(std::string(operation) +
                           " is a GPU placeholder and has no kernels yet");
}

void DERGpuBackend::requireGPU(runtime::ConstDofView view,
                               const char* operation)
{
  if (!view.isGPU()) {
    throw std::runtime_error(std::string(operation) +
                             " requires a GPU DofView");
  }
  if (!view.gpu()) {
    throw std::runtime_error(std::string(operation) +
                             " received an empty GPU DofView");
  }
}

void DERGpuBackend::requireGPU(runtime::DofView view,
                               const char* operation)
{
  requireGPU(view.asConst(), operation);
}

void DERGpuBackend::requireGPU(runtime::ConstGeometryView view,
                               const char* operation)
{
  if (!view.isGPU()) {
    throw std::runtime_error(std::string(operation) +
                             " requires a GPU GeometryView");
  }
  if (!view.gpu()) {
    throw std::runtime_error(std::string(operation) +
                             " received an empty GPU GeometryView");
  }
}

void DERGpuBackend::requireGPU(runtime::GeometryView view,
                               const char* operation)
{
  requireGPU(view.asConst(), operation);
}

void DERGpuBackend::requireSameDevice(runtime::ConstDofView lhs,
                                      runtime::ConstDofView rhs,
                                      const char* operation)
{
  if (lhs.device() != rhs.device()) {
    throw std::runtime_error(std::string(operation) +
                             " received buffers on different devices");
  }
}

void DERGpuBackend::requireSameDevice(runtime::ConstDofView lhs,
                                      runtime::ConstGeometryView rhs,
                                      const char* operation)
{
  if (lhs.device() != rhs.device()) {
    throw std::runtime_error(std::string(operation) +
                             " received buffers on different devices");
  }
}

}  // namespace ksk::der
