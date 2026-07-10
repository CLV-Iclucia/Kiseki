//
// Created by creeper on 23-8-13.
//

#pragma once
#include <Core/core.h>
namespace ksk {
namespace core {
class Frame final {
public:
  Index idx = 0;
  Real dt = 1.0 / 60.0;
  [[nodiscard]] Real currentTime() const { return idx * dt; }
  void onAdvance() { idx++; }
};

class Animation {
public:
  virtual ~Animation() = default;
  virtual void step(Frame &frame) = 0;
};
}
}
