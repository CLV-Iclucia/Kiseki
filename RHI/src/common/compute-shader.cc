//
// compute-shader.cc
//

#include <RHI/compute-shader.h>

namespace ksk::rhi {

void ComputeShaderBase::initialize(Device& device,
                                   ComputeShaderDefinition definition,
                                   ShaderCompileOptions options) {
  m_source = std::move(definition.source);
  m_pipeline = device.resolveTypedComputePipeline(m_source, options);
}

}  // namespace ksk::rhi
