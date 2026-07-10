//
// Created by creeper on 6/20/24.
//

#ifndef KISEKI_SPATIFY_INCLUDE_SPATIFY_PROPERTIES_H_
#define KISEKI_SPATIFY_INCLUDE_SPATIFY_PROPERTIES_H_
namespace spatify {

struct NonCopyable {
  NonCopyable() = default;
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;
};
}
#endif //KISEKI_SPATIFY_INCLUDE_SPATIFY_PROPERTIES_H_
