//
// Created by creeper on 5/23/24.
//

#ifndef KISEKI_SPATIFY_INCLUDE_SPATIFY_LBVH_H_
#define KISEKI_SPATIFY_INCLUDE_SPATIFY_LBVH_H_
#include <array>
#include <vector>
#include <concepts>
#include <atomic>
#include <Spatify/platform.h>
#include <Spatify/spatial-query.h>
#include <Spatify/parallel.h>
#include <Spatify/mortons.h>
namespace spatify {

inline int lcp(uint64_t a, uint64_t b) {
  return countLeadingZeros64Bit(a ^ b);
}

template <typename T>
concept BvhPrimitiveAccessor = requires(T t, int idx) {
  requires std::is_scalar_v<typename T::CoordType>;
  { t.bbox(idx) } -> std::convertible_to<BBox<typename T::CoordType, 3>>;
  { t.size() } -> std::convertible_to<int>;
};

template<typename T>
class LBVH {
 public:
  template<SpatialQuery Query, typename BBoxQuery>
  void runSpatialQuery(Query&& query, BBoxQuery&& bbox_query) const {
    std::array<int, 64> stack{};
    int top{};
    int nodeIdx = 0;
    if (!bbox_query(bbox[nodeIdx]))
      return;
    bool hit = false;
    while (true) {
      if (isLeaf(nodeIdx)) {
        int pr_id = idx[nodeIdx - nPrs + 1];
        bool flag = query(pr_id);
        hit |= flag;
        if (!top) return;
        nodeIdx = stack[--top];
        continue;
      }
      int lc = lch[nodeIdx];
      int rc = rch[nodeIdx];
      bool intl = bbox_query(bbox[lc]);
      bool intr = bbox_query(bbox[rc]);
      if (!intl && !intr) {
        if (!top) return;
        nodeIdx = stack[--top];
        continue;
      }
      if (intl && !intr) nodeIdx = lc;
      else if (!intl) nodeIdx = rc;
      else {
        nodeIdx = lc;
        stack[top++] = rc;
      }
    }
  }

  [[nodiscard]] int primitiveIndex(int nodeIdx) const {
    assert(nodeIdx >= nPrs - 1);
    assert(nodeIdx < 2 * nPrs - 1);
    return idx[nodeIdx - nPrs + 1];
  }

  [[nodiscard]] bool isLeaf(int index) const {
    return index >= nPrs - 1;
  }

  [[nodiscard]] int findSplit(int l, int r) const {
    if (mortons[l] == mortons[r])
      return (l + r) >> 1;
    int commonPrefix = lcp(mortons[l], mortons[r]);
    int search = l;
    int step = r - l;
    do {
      step = (step + 1) >> 1;
      if (int newSearch = search + step; newSearch < r) {
        uint64_t splitCode = mortons[newSearch];
        if (lcp(mortons[l], splitCode) > commonPrefix)
          search = newSearch;
      }
    } while (step > 1);
    return search;
  }

  [[nodiscard]] int delta(int i, int j) const {
    if (j < 0 || j > nPrs - 1) return -1;
    return lcp(mortons[i], mortons[j]);
  }

  uint32_t mortonCode(const Vector<T, 3> &p) const {
    int x = std::max((p.x - sceneBound.lo.x) / (sceneBound.hi.x - sceneBound.lo.x) * 1024, 0.0);
    int y = std::max((p.y - sceneBound.lo.y) / (sceneBound.hi.y - sceneBound.lo.y) * 1024, 0.0);
    int z = std::max((p.z - sceneBound.lo.z) / (sceneBound.hi.z - sceneBound.lo.z) * 1024, 0.0);
    return encodeMorton10bit(x, y, z);
  }

  uint64_t encodeBBox(const BBox<T, 3> &aabb, int index) const {
    uint32_t morton = mortonCode(aabb.centre());
    return (static_cast<uint64_t>(morton) << 32) | index;
  }
  int nPrs{};
  std::vector<BBox<Real, 3>> bbox{};
  template <BvhPrimitiveAccessor Accessor>
  void update(Accessor accessor) {
    nPrs = accessor.size();
    mortons.resize(2 * nPrs - 1);
    mortonsCopy.resize(2 * nPrs - 1);
    bbox.resize(2 * nPrs - 1);
    fa.resize(2 * nPrs - 1);
    lch.resize(nPrs - 1);
    rch.resize(nPrs - 1);
    idx.resize(nPrs);
    for (int i = 0; i < nPrs; i++) {
      bbox[i] = accessor.bbox(i);
      sceneBound.expand(bbox[i]);
    }
    tbb::parallel_for(0,
                 nPrs,
                 [this](int i) {
                   mortons[i] = encodeBBox(bbox[i], i);
                   idx[i] = i;
                 });
    mortonsCopy = mortons;
    tbb::parallel_sort(idx.begin(),
                  idx.end(),
                  [this](int a, int b) {
                    return mortons[a] < mortons[b];
                  });
    tbb::parallel_for(0,
                 nPrs,
                 [this](int i) {
                   mortons[i] = mortonsCopy[idx[i]];
                 });
    fa[0] = -1;
    tbb::parallel_for(0,
                 nPrs - 1,
                 [this](int i) {
                   int dir = delta(i, i + 1) > delta(i, i - 1) ? 1 : -1;
                   int min_delta = delta(i, i - dir);
                   int lmax = 2;
                   while (delta(i, i + lmax * dir) > min_delta) lmax <<= 1;
                   int len = 0;
                   for (int t = lmax >> 1; t; t >>= 1) {
                     if (delta(i, i + (len | t) * dir) > min_delta)
                       len |= t;
                   }
                   int l = std::min(i, i + len * dir);
                   int r = std::max(i, i + len * dir);
                   int split = findSplit(l, r);
                   if (l == split)
                     lch[i] = nPrs - 1 + split;
                   else lch[i] = split;
                   if (r == split + 1)
                     rch[i] = nPrs + split;
                   else rch[i] = split + 1;
                   fa[rch[i]] = fa[lch[i]] = i;
                 });
    tbb::parallel_for(0,
                 nPrs,
                 [this, &accessor](int i) {
                   int node_idx = nPrs + i - 1;
                   bbox[node_idx] = accessor.bbox(idx[i]);
                 });
    std::vector<std::atomic<bool>> processed(nPrs - 1);
    tbb::parallel_for(0,
                 nPrs,
                 [this, &processed](int i) {
                   int node_idx = nPrs + i - 1;
                   while (fa[node_idx] != -1) {
                     int parent = fa[node_idx];
                     if (!processed[parent].exchange(true)) return;
                     bbox[parent] = bbox[lch[parent]];
                     bbox[parent].expand(bbox[rch[parent]]);
                     node_idx = parent;
                   }
                 });
  }
 protected:
  std::vector<uint64_t> mortons{}, mortonsCopy{};
  std::vector<int> fa{};
  std::vector<int> lch{};
  std::vector<int> rch{};
  std::vector<int> idx{};
  BBox<Real, 3> sceneBound{};
};
}
#endif //KISEKI_SPATIFY_INCLUDE_SPATIFY_LBVH_H_