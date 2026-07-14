#include <DER/gpu/der-gpu-backend.h>

#include <DER/der-subsystem.h>

#include <stdexcept>
#include <string>

namespace ksk::der {

DERGpuBackend::DERGpuBackend(DERSubsystem& subsystem)
    : subsystem_(subsystem)
{
}

void DERGpuBackend::writeState(runtime::DofBuffer& q,
                               runtime::DofBuffer& qdot) const
{
  requireGPU(q, "DERGpuBackend::writeState");
  requireGPU(qdot, "DERGpuBackend::writeState");
  requireSameDevice(q, qdot, "DERGpuBackend::writeState");
  unsupported("DERGpuBackend::writeState");
}

void DERGpuBackend::readState(runtime::DofBuffer& q,
                              runtime::DofBuffer& qdot)
{
  requireGPU(q, "DERGpuBackend::readState");
  requireGPU(qdot, "DERGpuBackend::readState");
  requireSameDevice(q, qdot, "DERGpuBackend::readState");
  unsupported("DERGpuBackend::readState");
}

void DERGpuBackend::beginStep(const runtime::DofBuffer& q,
                              const runtime::DofBuffer& qdot,
                              double dt)
{
  (void)dt;
  requireGPU(q, "DERGpuBackend::beginStep");
  requireGPU(qdot, "DERGpuBackend::beginStep");
  requireSameDevice(q, qdot, "DERGpuBackend::beginStep");
  unsupported("DERGpuBackend::beginStep");
}

void DERGpuBackend::acceptStep(const runtime::DofBuffer& q,
                               runtime::DofBuffer& qdot,
                               double dt)
{
  (void)dt;
  requireGPU(q, "DERGpuBackend::acceptStep");
  requireGPU(qdot, "DERGpuBackend::acceptStep");
  requireSameDevice(q, qdot, "DERGpuBackend::acceptStep");
  unsupported("DERGpuBackend::acceptStep");
}

double DERGpuBackend::evaluateObjective(const runtime::DofBuffer& q,
                                        const runtime::DofBuffer& qdot,
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

void DERGpuBackend::assembleLocalGradient(runtime::DofBuffer& g) const
{
  requireGPU(g, "DERGpuBackend::assembleLocalGradient");
  unsupported("DERGpuBackend::assembleLocalGradient");
}

void DERGpuBackend::applyLocalMatrix(const runtime::DofBuffer& x,
                                     runtime::DofBuffer& y) const
{
  requireGPU(x, "DERGpuBackend::applyLocalMatrix");
  requireGPU(y, "DERGpuBackend::applyLocalMatrix");
  requireSameDevice(x, y, "DERGpuBackend::applyLocalMatrix");
  unsupported("DERGpuBackend::applyLocalMatrix");
}

void DERGpuBackend::solveLocalSystem(const runtime::DofBuffer& b,
                                     runtime::DofBuffer& x) const
{
  requireGPU(b, "DERGpuBackend::solveLocalSystem");
  requireGPU(x, "DERGpuBackend::solveLocalSystem");
  requireSameDevice(b, x, "DERGpuBackend::solveLocalSystem");
  unsupported("DERGpuBackend::solveLocalSystem");
}

void DERGpuBackend::mapDirectionToGeometry(const runtime::DofBuffer& dq,
                                           runtime::GeometryBuffer& dx) const
{
  requireGPU(dq, "DERGpuBackend::mapDirectionToGeometry");
  requireGPU(dx, "DERGpuBackend::mapDirectionToGeometry");
  requireSameDevice(dq, dx, "DERGpuBackend::mapDirectionToGeometry");
  unsupported("DERGpuBackend::mapDirectionToGeometry");
}

void DERGpuBackend::scatterContactGradient(
    std::span<const runtime::GeometryPointId> points,
    const runtime::GeometryBuffer& pointGradient,
    runtime::DofBuffer& g) const
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

void DERGpuBackend::applyContactHessian(const runtime::DofBuffer& dq,
                                        const runtime::ContactTable& contacts,
                                        runtime::DofBuffer& y) const
{
  (void)contacts;
  requireGPU(dq, "DERGpuBackend::applyContactHessian");
  requireGPU(y, "DERGpuBackend::applyContactHessian");
  requireSameDevice(dq, y, "DERGpuBackend::applyContactHessian");
  unsupported("DERGpuBackend::applyContactHessian");
}

void DERGpuBackend::unsupported(const char* operation)
{
  throw std::runtime_error(std::string(operation) +
                           " is a GPU placeholder and has no kernels yet");
}

void DERGpuBackend::requireGPU(const runtime::DofBuffer& buffer,
                               const char* operation)
{
  if (!buffer.isGPU()) {
    throw std::runtime_error(std::string(operation) +
                             " requires a GPU DofBuffer");
  }
  if (!buffer.gpu()) {
    throw std::runtime_error(std::string(operation) +
                             " received an empty GPU DofBuffer");
  }
}

void DERGpuBackend::requireGPU(const runtime::GeometryBuffer& buffer,
                               const char* operation)
{
  if (!buffer.isGPU()) {
    throw std::runtime_error(std::string(operation) +
                             " requires a GPU GeometryBuffer");
  }
  if (!buffer.gpu()) {
    throw std::runtime_error(std::string(operation) +
                             " received an empty GPU GeometryBuffer");
  }
}

void DERGpuBackend::requireSameDevice(const runtime::DofBuffer& lhs,
                                      const runtime::DofBuffer& rhs,
                                      const char* operation)
{
  if (lhs.device() != rhs.device()) {
    throw std::runtime_error(std::string(operation) +
                             " received buffers on different devices");
  }
}

void DERGpuBackend::requireSameDevice(const runtime::DofBuffer& lhs,
                                      const runtime::GeometryBuffer& rhs,
                                      const char* operation)
{
  if (lhs.device() != rhs.device()) {
    throw std::runtime_error(std::string(operation) +
                             " received buffers on different devices");
  }
}

}  // namespace ksk::der
