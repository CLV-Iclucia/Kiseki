// ============================================================================
// bvh-traverse.hlsli — shared helpers for LBVH broad-phase queries.
// ============================================================================
#ifndef BVH_TRAVERSE_HLSLI
#define BVH_TRAVERSE_HLSLI

// AABB overlap (inclusive). a = query (dilated), b = node/leaf (raw).
bool boxOverlap(double3 alo, double3 ahi, double3 blo, double3 bhi) {
    return alo.x <= bhi.x && ahi.x >= blo.x
        && alo.y <= bhi.y && ahi.y >= blo.y
        && alo.z <= bhi.z && ahi.z >= blo.z;
}

#endif // BVH_TRAVERSE_HLSLI
