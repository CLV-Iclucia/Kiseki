// ============================================================================
// lbvh-internal.hlsl — Karras (2012) radix-tree internal node construction.
// One thread per internal node i in [0, nPrs-1). Mirrors spatify::LBVH::update
// hierarchy step exactly. The CPU uses a 64-bit key (morton<<32 | origIdx); here
// the same lcp is reproduced with two 32-bit clz (morton in the high bits, the
// original index in the low bits), avoiding uint64.
// ============================================================================
[[vk::binding(0, 0)]] RWStructuredBuffer<int>  lch     : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<int>  rch     : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<int>  fa      : register(u2);
[[vk::binding(3, 0)]] StructuredBuffer<uint>   mortons : register(t0);  // sorted
[[vk::binding(4, 0)]] StructuredBuffer<uint>   idxs    : register(t1);  // sorted orig idx

struct PC { uint n; };   // n = nPrs
[[vk::push_constant]] PC pc;

int clz32(uint v) { return v == 0u ? 32 : (31 - (int)firstbithigh(v)); }

// lcp of the full 64-bit augmented key at sorted positions a,b (assumed valid).
// key = (morton << 32) | origIdx ; morton occupies the high 32 bits.
int lcpAt(int a, int b) {
    uint ma = mortons[a], mb = mortons[b];
    if (ma != mb) return clz32(ma ^ mb);
    return 32 + clz32(idxs[a] ^ idxs[b]);
}
int delta(int i, int j) {
    if (j < 0 || j > (int)pc.n - 1) return -1;
    return lcpAt(i, j);
}
bool keyEqual(int l, int r) { return mortons[l] == mortons[r] && idxs[l] == idxs[r]; }

int findSplit(int l, int r) {
    if (keyEqual(l, r)) return (l + r) >> 1;
    int commonPrefix = lcpAt(l, r);
    int search = l;
    int step = r - l;
    do {
        step = (step + 1) >> 1;
        int newSearch = search + step;
        if (newSearch < r) {
            if (lcpAt(l, newSearch) > commonPrefix) search = newSearch;
        }
    } while (step > 1);
    return search;
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint ii = tid.x;
    if (ii >= pc.n - 1u) return;
    int i = (int)ii;
    if (i == 0) fa[0] = -1;

    int dir = (delta(i, i + 1) > delta(i, i - 1)) ? 1 : -1;
    int minDelta = delta(i, i - dir);
    int lmax = 2;
    while (delta(i, i + lmax * dir) > minDelta) lmax <<= 1;
    int len = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (delta(i, i + (len | t) * dir) > minDelta) len |= t;
    }
    int l = min(i, i + len * dir);
    int r = max(i, i + len * dir);
    int split = findSplit(l, r);

    int lc = (l == split)     ? ((int)pc.n - 1 + split) : split;
    int rc = (r == split + 1) ? ((int)pc.n + split)     : (split + 1);
    lch[i] = lc;
    rch[i] = rc;
    fa[lc] = i;
    fa[rc] = i;
}
