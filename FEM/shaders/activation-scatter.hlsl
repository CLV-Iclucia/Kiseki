// ============================================================================
// activation-scatter.hlsl — module C, final gather.
// After the combined stream is stably sorted by kind key, the first `total`
// entries are the active constraints in CPU's PP|PE|PT|EE bucket order (with
// VT-before-EE within each bucket, preserved by the stable sort). Gather their
// indices into the compact output array.
//   outIdx[s*4 + k] = idxTmp[ valSorted[s]*4 + k ]
// ============================================================================
[[vk::binding(0, 0)]] RWStructuredBuffer<int>  outIdx    : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<uint>   valSorted : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<int>    idxTmp    : register(t1);

struct PC { uint total; };
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint s = tid.x;
    if (s >= pc.total) return;
    uint e = valSorted[s];
    outIdx[s * 4 + 0] = idxTmp[e * 4 + 0];
    outIdx[s * 4 + 1] = idxTmp[e * 4 + 1];
    outIdx[s * 4 + 2] = idxTmp[e * 4 + 2];
    outIdx[s * 4 + 3] = idxTmp[e * 4 + 3];
}
