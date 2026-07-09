#pragma once

#include <RHI/rhi.h>

namespace fluid::gpu {

#define FLUIDSIM_COMPUTE_SHADER_BEGIN(Type)                                  \
    class Type final : public sim::rhi::ComputeShader<Type> {                \
    public:                                                                   \
        DECLARE_COMPUTE_SHADER(Type);

#define FLUIDSIM_COMPUTE_SHADER_END() };

FLUIDSIM_COMPUTE_SHADER_BEGIN(P2GScatterCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, uGrid);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, vGrid);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, wGrid);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, uWeights);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, vWeights);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, wWeights);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, positions);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, velocities);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
        SHADER_PARAM_SCALAR(float, gridSpacing);
        SHADER_PARAM_SCALAR(float, originX);
        SHADER_PARAM_SCALAR(float, originY);
        SHADER_PARAM_SCALAR(float, originZ);
        SHADER_PARAM_SCALAR(uint32_t, numParticles);
        SHADER_PARAM_SCALAR(float, dt);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(P2GNormalizeCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, uGrid);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, vGrid);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, wGrid);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, uWeights);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, vWeights);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, wWeights);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, uValid);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, vValid);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, wValid);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
        SHADER_PARAM_SCALAR(uint32_t, maxFaces);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(G2PGatherCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, positions);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, velocities);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, uGrid);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, vGrid);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, wGrid);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
        SHADER_PARAM_SCALAR(float, gridSpacing);
        SHADER_PARAM_SCALAR(float, originX);
        SHADER_PARAM_SCALAR(float, originY);
        SHADER_PARAM_SCALAR(float, originZ);
        SHADER_PARAM_SCALAR(uint32_t, numParticles);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(AdvectCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, positions);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, velocities);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, uGrid);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, vGrid);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, wGrid);
        SHADER_PARAM_IMAGE (sim::rhi::ImageBinding, colliderSdf);
        SHADER_PARAM_SAMPLER(sim::rhi::SamplerRef, sdfSampler);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
        SHADER_PARAM_SCALAR(float, gridSpacing);
        SHADER_PARAM_SCALAR(float, originX);
        SHADER_PARAM_SCALAR(float, originY);
        SHADER_PARAM_SCALAR(float, originZ);
        SHADER_PARAM_SCALAR(uint32_t, numParticles);
        SHADER_PARAM_SCALAR(float, dt);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(CflReduceCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, velocities);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, partialMax);
        SHADER_PARAM_SCALAR(uint32_t, numParticles);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(CflReduceFinalCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, partialMax);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, scalarOut);
        SHADER_PARAM_SCALAR(uint32_t, numGroups);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(BodyForceCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, vGrid);
        SHADER_PARAM_SCALAR(float, dt);
        SHADER_PARAM_SCALAR(float, gravityY);
        SHADER_PARAM_SCALAR(uint32_t, numVFaces);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(ExtrapolateCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, gridIn);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, gridOut);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, validIn);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, validOut);
        SHADER_PARAM_SCALAR(uint32_t, faceResX);
        SHADER_PARAM_SCALAR(uint32_t, faceResY);
        SHADER_PARAM_SCALAR(uint32_t, faceResZ);
        SHADER_PARAM_SCALAR(uint32_t, numFaces);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(DirichletCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, uGrid);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, vGrid);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, wGrid);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(ColliderCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, uGrid);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, vGrid);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, wGrid);
        SHADER_PARAM_IMAGE (sim::rhi::ImageBinding, colliderSdf);
        SHADER_PARAM_SAMPLER(sim::rhi::SamplerRef, sdfSampler);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
        SHADER_PARAM_SCALAR(float, gridSpacing);
        SHADER_PARAM_SCALAR(float, originX);
        SHADER_PARAM_SCALAR(float, originY);
        SHADER_PARAM_SCALAR(float, originZ);
        SHADER_PARAM_SCALAR(uint32_t, maxFaces);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(BuildWeightsCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, faceWeightsU);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, faceWeightsV);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, faceWeightsW);
        SHADER_PARAM_IMAGE (sim::rhi::ImageBinding, colliderSdf);
        SHADER_PARAM_SAMPLER(sim::rhi::SamplerRef, sdfSampler);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
        SHADER_PARAM_SCALAR(float, gridSpacing);
        SHADER_PARAM_SCALAR(float, originX);
        SHADER_PARAM_SCALAR(float, originY);
        SHADER_PARAM_SCALAR(float, originZ);
        SHADER_PARAM_SCALAR(uint32_t, maxFaces);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(BuildSystemCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, Adiag);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, Aneighbour0);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, Aneighbour1);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, Aneighbour2);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, Aneighbour3);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, Aneighbour4);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, Aneighbour5);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, rhs);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, active);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, uGrid);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, vGrid);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, wGrid);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, faceWeightsU);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, faceWeightsV);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, faceWeightsW);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, uValid);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, vValid);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, wValid);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
        SHADER_PARAM_SCALAR(float, gridSpacing);
        SHADER_PARAM_SCALAR(float, density);
        SHADER_PARAM_SCALAR(float, dt);
        SHADER_PARAM_SCALAR(float, originX);
        SHADER_PARAM_SCALAR(float, originY);
        SHADER_PARAM_SCALAR(float, originZ);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(ProjectCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, uGrid);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, vGrid);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, wGrid);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, pressure);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, faceWeightsU);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, faceWeightsV);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, faceWeightsW);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
        SHADER_PARAM_SCALAR(float, gridSpacing);
        SHADER_PARAM_SCALAR(float, density);
        SHADER_PARAM_SCALAR(float, dt);
        SHADER_PARAM_SCALAR(uint32_t, maxFaces);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(ReconstructSdfCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::ImageRef, fluidSdf);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, positions);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
        SHADER_PARAM_SCALAR(float, gridSpacing);
        SHADER_PARAM_SCALAR(float, originX);
        SHADER_PARAM_SCALAR(float, originY);
        SHADER_PARAM_SCALAR(float, originZ);
        SHADER_PARAM_SCALAR(uint32_t, numParticles);
        SHADER_PARAM_SCALAR(float, particleRadius);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(SmoothSdfCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_IMAGE (sim::rhi::ImageBinding, srcSdf);
        SHADER_PARAM_UAV   (sim::rhi::ImageRef, dstSdf);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(JacobiIterCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, pressureIn);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, pressureOut);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, Adiag);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour0);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour1);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour2);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour3);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour4);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour5);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, rhs);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
        SHADER_PARAM_SCALAR(uint32_t, numCells);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(PCGSpMVCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, output);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, Adiag);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour0);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour1);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour2);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour3);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour4);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour5);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, src);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
        SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
        SHADER_PARAM_SCALAR(uint32_t, numCells);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(PCGDotCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, vecA);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, vecB);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, reduceBuf);
        SHADER_PARAM_SCALAR(uint32_t, numCells);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(PCGReduceFinalCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, reduceBuf);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, scalarOut);
        SHADER_PARAM_SCALAR(uint32_t, numGroups);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(PCGSaxpyCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, y);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, x);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, alphaBuf);
        SHADER_PARAM_SCALAR(uint32_t, numCells);
        SHADER_PARAM_SCALAR(float, sign);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(PCGJacobiPrecondCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, z);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, r);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, Adiag);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, active);
        SHADER_PARAM_SCALAR(uint32_t, numCells);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(PCGScalarDivCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, num);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, denom);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, result);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

FLUIDSIM_COMPUTE_SHADER_BEGIN(PCGUpdateSCS)
    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, s);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, z);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, sigmaNewBuf);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, sigmaOldBuf);
        SHADER_PARAM_SCALAR(uint32_t, numCells);
    SHADER_PARAMS_END();
FLUIDSIM_COMPUTE_SHADER_END()

#undef FLUIDSIM_COMPUTE_SHADER_BEGIN
#undef FLUIDSIM_COMPUTE_SHADER_END

} // namespace fluid::gpu
