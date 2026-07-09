#ifndef SIMCRAFT_RHI_STRUCTURED_BUFFER_ACCESS_HLSLI
#define SIMCRAFT_RHI_STRUCTURED_BUFFER_ACCESS_HLSLI

float3 load_float3(StructuredBuffer<float> src, uint index) {
    const uint base = index * 3u;
    return float3(src[base], src[base + 1u], src[base + 2u]);
}

float3 load_float3(RWStructuredBuffer<float> src, uint index) {
    const uint base = index * 3u;
    return float3(src[base], src[base + 1u], src[base + 2u]);
}

void store_float3(
    RWStructuredBuffer<float> dst, uint index, float3 value) {
    const uint base = index * 3u;
    dst[base] = value.x;
    dst[base + 1u] = value.y;
    dst[base + 2u] = value.z;
}

double3 load_double3(StructuredBuffer<double> src, uint index) {
    const uint base = index * 3u;
    return double3(src[base], src[base + 1u], src[base + 2u]);
}

double3 load_double3(RWStructuredBuffer<double> src, uint index) {
    const uint base = index * 3u;
    return double3(src[base], src[base + 1u], src[base + 2u]);
}

void store_double3(
    RWStructuredBuffer<double> dst, uint index, double3 value) {
    const uint base = index * 3u;
    dst[base] = value.x;
    dst[base + 1u] = value.y;
    dst[base + 2u] = value.z;
}

float3x3 load_float3x3_col_major(
    StructuredBuffer<float> src, uint index) {
    const uint base = index * 9u;
    float3x3 value;
    [unroll]
    for (uint column = 0u; column < 3u; ++column) {
        [unroll]
        for (uint row = 0u; row < 3u; ++row) {
            value[row][column] = src[base + column * 3u + row];
        }
    }
    return value;
}

float3x3 load_float3x3_col_major(
    RWStructuredBuffer<float> src, uint index) {
    const uint base = index * 9u;
    float3x3 value;
    [unroll]
    for (uint column = 0u; column < 3u; ++column) {
        [unroll]
        for (uint row = 0u; row < 3u; ++row) {
            value[row][column] = src[base + column * 3u + row];
        }
    }
    return value;
}

void store_float3x3_col_major(
    RWStructuredBuffer<float> dst, uint index, float3x3 value) {
    const uint base = index * 9u;
    [unroll]
    for (uint column = 0u; column < 3u; ++column) {
        [unroll]
        for (uint row = 0u; row < 3u; ++row) {
            dst[base + column * 3u + row] = value[row][column];
        }
    }
}

double3x3 load_double3x3_col_major(
    StructuredBuffer<double> src, uint index) {
    const uint base = index * 9u;
    double3x3 value;
    [unroll]
    for (uint column = 0u; column < 3u; ++column) {
        [unroll]
        for (uint row = 0u; row < 3u; ++row) {
            value[row][column] = src[base + column * 3u + row];
        }
    }
    return value;
}

double3x3 load_double3x3_col_major(
    RWStructuredBuffer<double> src, uint index) {
    const uint base = index * 9u;
    double3x3 value;
    [unroll]
    for (uint column = 0u; column < 3u; ++column) {
        [unroll]
        for (uint row = 0u; row < 3u; ++row) {
            value[row][column] = src[base + column * 3u + row];
        }
    }
    return value;
}

void store_double3x3_col_major(
    RWStructuredBuffer<double> dst, uint index, double3x3 value) {
    const uint base = index * 9u;
    [unroll]
    for (uint column = 0u; column < 3u; ++column) {
        [unroll]
        for (uint row = 0u; row < 3u; ++row) {
            dst[base + column * 3u + row] = value[row][column];
        }
    }
}

float3 load_xyz(StructuredBuffer<float4> src, uint index) {
    return src[index].xyz;
}

void store_xyz_preserve_w(
    RWStructuredBuffer<float4> dst, uint index, float3 value) {
    float4 stored = dst[index];
    stored.xyz = value;
    dst[index] = stored;
}

double3 load_xyz(StructuredBuffer<double4> src, uint index) {
    return src[index].xyz;
}

void store_xyz_preserve_w(
    RWStructuredBuffer<double4> dst, uint index, double3 value) {
    double4 stored = dst[index];
    stored.xyz = value;
    dst[index] = stored;
}

#endif
