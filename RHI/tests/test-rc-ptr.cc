//
// test-rc-ptr.cc
// Pure CPU test for ksk::rhi::RcPtr<T>. No Vulkan device needed.
//

#include <RHI/rc-ptr.h>

#include <atomic>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using ksk::rhi::RcPtr;

namespace {

// Mock target satisfying the RcPtr<T> contract: addRef/release noexcept.
struct MockRefCounted {
  std::atomic<uint32_t> rc{0};
  std::atomic<int>* destroyCounter = nullptr;

  void addRef() noexcept { rc.fetch_add(1, std::memory_order_relaxed); }
  void release() noexcept {
    if (rc.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      if (destroyCounter) destroyCounter->fetch_add(1, std::memory_order_relaxed);
      delete this;
    }
  }
};

}  // namespace

TEST(RcPtrTest, DefaultIsNull) {
  RcPtr<MockRefCounted> p;
  EXPECT_FALSE(p.valid());
  EXPECT_FALSE(static_cast<bool>(p));
  EXPECT_EQ(p.get(), nullptr);
}

TEST(RcPtrTest, ConstructIncrementsRefcount) {
  std::atomic<int> destroyed{0};
  auto* raw = new MockRefCounted{};
  raw->destroyCounter = &destroyed;
  {
    RcPtr<MockRefCounted> p(raw);
    EXPECT_TRUE(p.valid());
    EXPECT_EQ(p->rc.load(), 1u);
    EXPECT_EQ(destroyed.load(), 0);
  }
  EXPECT_EQ(destroyed.load(), 1);
}

TEST(RcPtrTest, AdoptDoesNotIncrement) {
  std::atomic<int> destroyed{0};
  auto* raw = new MockRefCounted{};
  raw->destroyCounter = &destroyed;
  raw->addRef();  // Backend ctor convention: refcount already 1 before adopt.
  {
    auto p = RcPtr<MockRefCounted>::adopt(raw);
    EXPECT_EQ(p->rc.load(), 1u);
  }
  EXPECT_EQ(destroyed.load(), 1);
}

// Regression: misusing `adopt` (without manually addRef'ing first) means
// m_rc stays at 0. The first release underflows and destroy() is never
// called → silent leak. We assert the underflow behavior here so callers
// who paste this pattern get reminded by the test suite (and so the bug
// that caused test-buffer's "leaked VkBuffer" doesn't regress).
TEST(RcPtrTest, AdoptWithoutAddRefLeaks) {
  std::atomic<int> destroyed{0};
  auto* raw = new MockRefCounted{};
  raw->destroyCounter = &destroyed;
  // ❌ INTENTIONALLY WRONG: adopt without prior addRef.
  {
    auto p = RcPtr<MockRefCounted>::adopt(raw);
    EXPECT_EQ(p->rc.load(), 0u);  // m_rc stays at 0
    // Out of scope → release → fetch_sub returns 0 → destroy NOT called.
  }
  EXPECT_EQ(destroyed.load(), 0);  // leaked!
  // Clean up so the test itself doesn't leak into other tests.
  delete raw;
}

TEST(RcPtrTest, CopyShares) {
  std::atomic<int> destroyed{0};
  auto* raw = new MockRefCounted{};
  raw->destroyCounter = &destroyed;
  {
    RcPtr<MockRefCounted> p(raw);
    {
      RcPtr<MockRefCounted> q = p;  // refcount becomes 2
      EXPECT_EQ(p->rc.load(), 2u);
      EXPECT_EQ(p.get(), q.get());
    }
    EXPECT_EQ(p->rc.load(), 1u);
  }
  EXPECT_EQ(destroyed.load(), 1);
}

TEST(RcPtrTest, MoveStealsOwnership) {
  std::atomic<int> destroyed{0};
  auto* raw = new MockRefCounted{};
  raw->destroyCounter = &destroyed;
  {
    RcPtr<MockRefCounted> p(raw);
    RcPtr<MockRefCounted> q = std::move(p);
    EXPECT_FALSE(p.valid());
    EXPECT_TRUE(q.valid());
    EXPECT_EQ(q->rc.load(), 1u);
  }
  EXPECT_EQ(destroyed.load(), 1);
}

TEST(RcPtrTest, AssignReleasesOldThenIncrementsNew) {
  std::atomic<int> destroyedA{0};
  std::atomic<int> destroyedB{0};
  auto* a = new MockRefCounted{};
  a->destroyCounter = &destroyedA;
  auto* b = new MockRefCounted{};
  b->destroyCounter = &destroyedB;
  {
    RcPtr<MockRefCounted> p(a);
    RcPtr<MockRefCounted> q(b);
    p = q;  // a freed, b refcount = 2
    EXPECT_EQ(destroyedA.load(), 1);
    EXPECT_EQ(b->rc.load(), 2u);
  }
  EXPECT_EQ(destroyedB.load(), 1);
}

TEST(RcPtrTest, ResetReleasesEarly) {
  std::atomic<int> destroyed{0};
  auto* raw = new MockRefCounted{};
  raw->destroyCounter = &destroyed;
  RcPtr<MockRefCounted> p(raw);
  p.reset();
  EXPECT_FALSE(p.valid());
  EXPECT_EQ(destroyed.load(), 1);
}

TEST(RcPtrTest, ConcurrentAddRefReleaseIsRace_Free) {
  // 8 threads each take 10000 copies and let them go out of scope. We expect
  // exactly one destroy at the end, not zero (leaked) and not >1 (UB).
  std::atomic<int> destroyed{0};
  auto* raw = new MockRefCounted{};
  raw->destroyCounter = &destroyed;
  RcPtr<MockRefCounted> root(raw);

  constexpr int kThreads = 8;
  constexpr int kIters = 10000;
  std::vector<std::thread> ts;
  ts.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    ts.emplace_back([&] {
      for (int j = 0; j < kIters; ++j) {
        RcPtr<MockRefCounted> copy = root;
        (void)copy.get();
      }
    });
  }
  for (auto& t : ts) t.join();

  EXPECT_TRUE(root.valid());
  EXPECT_EQ(root->rc.load(), 1u);
  EXPECT_EQ(destroyed.load(), 0);
  root.reset();
  EXPECT_EQ(destroyed.load(), 1);
}
