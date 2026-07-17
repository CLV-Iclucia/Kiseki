#include <FluidSim/gpu/gpu-shaders.h>

#ifndef FLUIDSIM_SHADER_DIR
#define FLUIDSIM_SHADER_DIR "."
#endif

namespace fluid::gpu {

IMPLEMENT_COMPUTE_SHADER(P2GScatterCS, FLUIDSIM_SHADER_DIR "/p2g-scatter.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(P2GNormalizeCS, FLUIDSIM_SHADER_DIR "/p2g-normalize.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(G2PGatherCS, FLUIDSIM_SHADER_DIR "/g2p-gather.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(AdvectCS, FLUIDSIM_SHADER_DIR "/advect.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(CflReduceCS, FLUIDSIM_SHADER_DIR "/cfl-reduce.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(CflReduceFinalCS, FLUIDSIM_SHADER_DIR "/cfl-reduce-final.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(BodyForceCS, FLUIDSIM_SHADER_DIR "/body-force.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(ExtrapolateCS, FLUIDSIM_SHADER_DIR "/extrapolate.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(DirichletCS, FLUIDSIM_SHADER_DIR "/dirichlet.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(ColliderCS, FLUIDSIM_SHADER_DIR "/collider.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(BuildWeightsCS, FLUIDSIM_SHADER_DIR "/build-weights.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(BuildSystemCS, FLUIDSIM_SHADER_DIR "/build-system.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(ProjectCS, FLUIDSIM_SHADER_DIR "/project.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(BuildParticleHashCS, FLUIDSIM_SHADER_DIR "/build-particle-hash.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(BuildParticleCellRangesCS, FLUIDSIM_SHADER_DIR "/build-particle-cell-ranges.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(ReconstructSdfHashedCS, FLUIDSIM_SHADER_DIR "/reconstruct-sdf-hashed.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(SmoothSdfCS, FLUIDSIM_SHADER_DIR "/smooth-sdf.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(MarchingCubesCS, FLUIDSIM_SHADER_DIR "/marching-cubes.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(JacobiIterCS, FLUIDSIM_SHADER_DIR "/jacobi.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(PCGSpMVCS, FLUIDSIM_SHADER_DIR "/cg-spmv.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(PCGDotCS, FLUIDSIM_SHADER_DIR "/cg-dot-product.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(PCGReduceFinalCS, FLUIDSIM_SHADER_DIR "/cg-reduce-final.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(PCGSaxpyCS, FLUIDSIM_SHADER_DIR "/cg-saxpy.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(PCGJacobiPrecondCS, FLUIDSIM_SHADER_DIR "/cg-jacobi.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(PCGScalarDivCS, FLUIDSIM_SHADER_DIR "/cg-scalar-div.hlsl", "main");
IMPLEMENT_COMPUTE_SHADER(PCGUpdateSCS, FLUIDSIM_SHADER_DIR "/cg-update-s.hlsl", "main");

} // namespace fluid::gpu
