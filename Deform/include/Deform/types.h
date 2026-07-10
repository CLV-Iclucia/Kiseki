//
// Created by creeper on 5/23/24.
//

#ifndef KISEKI_DEFORM_TYPES_H_
#define KISEKI_DEFORM_TYPES_H_
#include <Maths/types.h>
#include <Maths/tensor.h>
namespace ksk::deform {
template <typename T, int N, int M>
using Matrix = maths::Matrix<T, N, M>;
using maths::Vector;
using maths::Real;
}
#endif //KISEKI_DEFORM_TYPES_H_
