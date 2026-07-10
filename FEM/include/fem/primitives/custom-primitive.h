//
// Created by creeper on 10/22/24.
//

#ifndef KISEKI_FEM_INCLUDE_FEM_PRIMITIVES_CUSTOM_PRIMITIVE_H_
#define KISEKI_FEM_INCLUDE_FEM_PRIMITIVES_CUSTOM_PRIMITIVE_H_
#include <memory>
#include <fem/primitives/primitive-base.h>
namespace fem {
struct CustomPrimitiveImpl : PrimitiveBase {
  virtual void assembleEnergyGradient(VecXd &globalGrad) const = 0;
  virtual size_t dofDim() const = 0;
};

struct CustomPrimitive {
  std::unique_ptr<CustomPrimitiveImpl> custom{};
  void assembleEnergyGradient(VecXd &globalGrad) const {
    custom->assembleEnergyGradient(globalGrad);
  }
  [[nodiscard]] size_t dofDim() const {
    return custom->dofDim();
  }
};
}
#endif //KISEKI_FEM_INCLUDE_FEM_PRIMITIVES_CUSTOM_PRIMITIVE_H_
