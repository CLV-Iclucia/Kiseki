//
// Created by creeper on 5/28/24.
//

#pragma once

#include <map>
#include <type_traits>
namespace ksk::core {
#define REPLACE_FOR_EACH_1(what, _1) what(_1)
#define REPLACE_FOR_EACH_2(what, _1, _2) what(_1) what(_2)
#define REPLACE_FOR_EACH_3(what, _1, _2, _3) what(_1) what(_2) what(_3)
#define REPLACE_FOR_EACH_4(what, _1, _2, _3, _4)                               \
  what(_1) what(_2) what(_3) what(_4)
#define REPLACE_FOR_EACH_5(what, _1, _2, _3, _4, _5)                           \
  what(_1) what(_2) what(_3) what(_4) what(_5)
#define REPLACE_FOR_EACH_6(what, _1, _2, _3, _4, _5, _6)                       \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6)
#define REPLACE_FOR_EACH_7(what, _1, _2, _3, _4, _5, _6, _7)                   \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7)
#define REPLACE_FOR_EACH_8(what, _1, _2, _3, _4, _5, _6, _7, _8)               \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)
#define REPLACE_FOR_EACH_9(what, _1, _2, _3, _4, _5, _6, _7, _8, _9)           \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9)
#define REPLACE_FOR_EACH_10(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10)     \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10)
#define REPLACE_FOR_EACH_11(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11)                                               \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11)
#define REPLACE_FOR_EACH_12(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12)                                          \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12)
#define REPLACE_FOR_EACH_13(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13)                                     \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13)
#define REPLACE_FOR_EACH_14(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14)                                \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14)
#define REPLACE_FOR_EACH_15(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15)                           \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)
#define REPLACE_FOR_EACH_16(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15, _16)                      \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)     \
          what(_16)
#define REPLACE_FOR_EACH_17(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15, _16, _17)                 \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)     \
          what(_16) what(_17)
#define REPLACE_FOR_EACH_18(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15, _16, _17, _18)            \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)     \
          what(_16) what(_17) what(_18)
#define REPLACE_FOR_EACH_19(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15, _16, _17, _18, _19)       \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)     \
          what(_16) what(_17) what(_18) what(_19)
#define REPLACE_FOR_EACH_20(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15, _16, _17, _18, _19, _20)  \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)     \
          what(_16) what(_17) what(_18) what(_19) what(_20)
#define REPLACE_FOR_EACH_21(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15, _16, _17, _18, _19, _20,  \
                            _21)                                               \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)     \
          what(_16) what(_17) what(_18) what(_19) what(_20) what(_21)
#define REPLACE_FOR_EACH_22(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15, _16, _17, _18, _19, _20,  \
                            _21, _22)                                          \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)     \
          what(_16) what(_17) what(_18) what(_19) what(_20) what(_21)          \
              what(_22)
#define REPLACE_FOR_EACH_23(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15, _16, _17, _18, _19, _20,  \
                            _21, _22, _23)                                     \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)     \
          what(_16) what(_17) what(_18) what(_19) what(_20) what(_21)          \
              what(_22) what(_23)
#define REPLACE_FOR_EACH_24(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15, _16, _17, _18, _19, _20,  \
                            _21, _22, _23, _24)                                \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)     \
          what(_16) what(_17) what(_18) what(_19) what(_20) what(_21)          \
              what(_22) what(_23) what(_24)
#define REPLACE_FOR_EACH_25(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15, _16, _17, _18, _19, _20,  \
                            _21, _22, _23, _24, _25)                           \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)     \
          what(_16) what(_17) what(_18) what(_19) what(_20) what(_21)          \
              what(_22) what(_23) what(_24) what(_25)
#define REPLACE_FOR_EACH_26(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15, _16, _17, _18, _19, _20,  \
                            _21, _22, _23, _24, _25, _26)                      \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)     \
          what(_16) what(_17) what(_18) what(_19) what(_20) what(_21)          \
              what(_22) what(_23) what(_24) what(_25) what(_26)
#define REPLACE_FOR_EACH_27(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15, _16, _17, _18, _19, _20,  \
                            _21, _22, _23, _24, _25, _26, _27)                 \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)     \
          what(_16) what(_17) what(_18) what(_19) what(_20) what(_21)          \
              what(_22) what(_23) what(_24) what(_25) what(_26) what(_27)
#define REPLACE_FOR_EACH_28(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15, _16, _17, _18, _19, _20,  \
                            _21, _22, _23, _24, _25, _26, _27, _28)            \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)     \
          what(_16) what(_17) what(_18) what(_19) what(_20) what(_21)          \
              what(_22) what(_23) what(_24) what(_25) what(_26) what(_27)      \
                  what(_28)
#define REPLACE_FOR_EACH_29(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15, _16, _17, _18, _19, _20,  \
                            _21, _22, _23, _24, _25, _26, _27, _28, _29)       \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)     \
          what(_16) what(_17) what(_18) what(_19) what(_20) what(_21)          \
              what(_22) what(_23) what(_24) what(_25) what(_26) what(_27)      \
                  what(_28) what(_29)
#define REPLACE_FOR_EACH_30(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15, _16, _17, _18, _19, _20,  \
                            _21, _22, _23, _24, _25, _26, _27, _28, _29, _30)  \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)     \
          what(_16) what(_17) what(_18) what(_19) what(_20) what(_21)          \
              what(_22) what(_23) what(_24) what(_25) what(_26) what(_27)      \
                  what(_28) what(_29) what(_30)
#define REPLACE_FOR_EACH_31(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15, _16, _17, _18, _19, _20,  \
                            _21, _22, _23, _24, _25, _26, _27, _28, _29, _30,  \
                            _31)                                               \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)     \
          what(_16) what(_17) what(_18) what(_19) what(_20) what(_21)          \
              what(_22) what(_23) what(_24) what(_25) what(_26) what(_27)      \
                  what(_28) what(_29) what(_30) what(_31)
#define REPLACE_FOR_EACH_32(what, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,     \
                            _11, _12, _13, _14, _15, _16, _17, _18, _19, _20,  \
                            _21, _22, _23, _24, _25, _26, _27, _28, _29, _30,  \
                            _31, _32)                                          \
  what(_1) what(_2) what(_3) what(_4) what(_5) what(_6) what(_7) what(_8)      \
      what(_9) what(_10) what(_11) what(_12) what(_13) what(_14) what(_15)     \
          what(_16) what(_17) what(_18) what(_19) what(_20) what(_21)          \
              what(_22) what(_23) what(_24) what(_25) what(_26) what(_27)      \
                  what(_28) what(_29) what(_30) what(_31) what(_32)

#define NARGS_SEQ(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, \
                  _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26,  \
                  _27, _28, _29, _30, _31, _32, N, ...)                        \
  N
#define NARGS(...)                                                             \
  NARGS_SEQ(__VA_ARGS__, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,   \
            19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

#define CONCAT(x, y) x##y
#define STRING(x) #x
#define CONCAT_EVAL(x, y) CONCAT(x, y)
#define REPLACE_FOR_EACH(what, ...)                                            \
  CONCAT_EVAL(REPLACE_FOR_EACH_, NARGS(__VA_ARGS__))(what, __VA_ARGS__)
#define REFLECTION_IMPL(name) func(#name, this->name);

template <typename T> constexpr bool can_reflect() {
  using decayed_t = std::decay_t<T>;
  if constexpr (requires { decayed_t::reflect::is_reflectable; }) {
    return true;
  } else
    return false;
}
} // namespace ksk::core

#define REFLECT(...)                                                        \
  struct reflect {                                                             \
    static constexpr bool is_reflectable = true;                               \
  };                                                                           \
  template <typename Func> constexpr void forEachMember(Func &&func) {  \
    REPLACE_FOR_EACH(REFLECTION_IMPL, __VA_ARGS__)                             \
  }
