//
// rc-ptr.h
// Intrusive ref-counted smart pointer for Rhi resource handles.
// See docs/rhi-plan.md §3.0 (R10–R14 / R21).
//
// Convention: T must provide noexcept methods
//     void addRef() noexcept;
//     void release() noexcept;
// (No concept constraint — substitution failure surfaces at first use.)
//

#pragma once

namespace ksk::rhi {

template <class T>
class RcPtr {
 public:
  RcPtr() = default;

  explicit RcPtr(T* p) : m_p(p) {
    if (m_p) m_p->addRef();
  }

  RcPtr(const RcPtr& o) : m_p(o.m_p) {
    if (m_p) m_p->addRef();
  }

  RcPtr(RcPtr&& o) noexcept : m_p(o.m_p) { o.m_p = nullptr; }

  ~RcPtr() {
    if (m_p) m_p->release();
  }

  RcPtr& operator=(const RcPtr& o) {
    if (this != &o) {
      if (o.m_p) o.m_p->addRef();
      if (m_p) m_p->release();
      m_p = o.m_p;
    }
    return *this;
  }

  RcPtr& operator=(RcPtr&& o) noexcept {
    if (this != &o) {
      if (m_p) m_p->release();
      m_p = o.m_p;
      o.m_p = nullptr;
    }
    return *this;
  }

  T* operator->() const noexcept { return m_p; }
  T& operator*() const noexcept { return *m_p; }
  T* get() const noexcept { return m_p; }

  explicit operator bool() const noexcept { return m_p != nullptr; }
  bool valid() const noexcept { return m_p != nullptr; }

  void reset() noexcept {
    if (m_p) {
      m_p->release();
      m_p = nullptr;
    }
  }

  // Take ownership of a raw pointer that ALREADY HAS refcount +1.
  //
  // ⚠ Most callers want `RcPtr<T>(raw)` — that ctor calls `addRef()` and
  // brings the count from 0 to 1. `adopt` is for the rare case where you
  // manually `addRef`'d (or otherwise already accounted for the +1) and
  // want to hand off without double-incrementing. Misusing it leaves
  // m_rc=0 forever, so the next release underflows and destroy() is never
  // called → silent leak.
  static RcPtr adopt(T* raw) noexcept {
    RcPtr r;
    r.m_p = raw;
    return r;
  }

  friend bool operator==(const RcPtr& a, const RcPtr& b) noexcept { return a.m_p == b.m_p; }
  friend bool operator!=(const RcPtr& a, const RcPtr& b) noexcept { return a.m_p != b.m_p; }

 private:
  T* m_p = nullptr;
};

}  // namespace ksk::rhi
