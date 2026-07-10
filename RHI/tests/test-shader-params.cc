//
// test-shader-params.cc
// R4 verification: SHADER_PARAMS macro system.
// See docs/rhi-plan.md §8.1 (test-shader-params.cc bullet) and §3.4.4 / §13.1.
//
// CPU-only: ParamSlot operator= behavior + per-instance schema build +
// _resolve happy path + error paths. The end-to-end saxpy GPU run lives in
// test-compute-trivial.cc.
//

#include <RHI/buffer.h>
#include <RHI/image.h>
#include <RHI/reflection.h>
#include <RHI/shader-params.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>

using namespace ksk::rhi;

// ---------- A typical SHADER_PARAMS block --------------------------------
//
// Note: SHADER_PARAMS classes can now live anywhere — function-local,
// namespace-local, or inside another class. The schema is built per-instance
// in the ctor via DMI side effects, so there are no static initialisers
// that need to run before main(). Keeping these at namespace scope here
// purely for legibility.

SHADER_PARAMS_BEGIN(SaxpyParams)
  SHADER_PARAM_UAV   (BufferRef, g_y);
  SHADER_PARAM_SCALAR(uint32_t,  count);
  SHADER_PARAM_SCALAR(float,     alpha);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(KitchenSinkParams)
  SHADER_PARAM_UAV    (BufferRef,    rwBuf);
  SHADER_PARAM_SRV    (BufferRef,    cb);
  SHADER_PARAM_IMAGE  (ImageBinding, tex);
  SHADER_PARAM_SAMPLER(SamplerRef,   smp);
  SHADER_PARAM_SCALAR (uint32_t,     n);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(NoScalarParams)
  SHADER_PARAM_UAV(BufferRef, g_y);
SHADER_PARAMS_END();

// ---------- Tests --------------------------------------------------------

// Verifies the wrapper behaviour: ParamSlot<T>'s operator= must look like a
// plain T assignment, and reading the slot via implicit conversion or get()
// must produce the assigned T value. This is the headline "field is a
// wrapper, but feels like a raw T" property the macro system promises.
TEST(ParamSlot, AssignmentLooksLikeRawT) {
  SaxpyParams p;
  // operator=(const float&) overload:
  p.alpha = 2.5f;
  // implicit conversion + explicit get() should both round-trip:
  EXPECT_FLOAT_EQ(static_cast<float>(p.alpha), 2.5f);
  EXPECT_FLOAT_EQ(p.alpha.get(), 2.5f);

  // Move-assignment of a count value:
  p.count = 1024u;
  EXPECT_EQ(static_cast<uint32_t>(p.count), 1024u);

  // Default-constructed slots: default T value.
  SaxpyParams q;
  EXPECT_FLOAT_EQ(q.alpha.get(), 0.0f);
  EXPECT_EQ(q.count.get(), 0u);
}

TEST(ShaderParamsSchema, BuiltPerInstanceInDeclarationOrder) {
  // Each instance gets its own _schema vector, populated by the SHADER_PARAM_*
  // DMI side effects on construction. Two distinct instances should hold the
  // same schema content (both built from the same field declarations).
  SaxpyParams a;
  SaxpyParams b;

  ASSERT_EQ(a._schema.size(), 3u);
  ASSERT_EQ(b._schema.size(), 3u);

  EXPECT_STREQ(a._schema[0].name, "g_y");
  EXPECT_EQ(a._schema[0].kind, detail::FieldKind::UAVBuffer);
  EXPECT_STREQ(a._schema[1].name, "count");
  EXPECT_EQ(a._schema[1].kind, detail::FieldKind::Scalar);
  EXPECT_EQ(a._schema[1].valueSize, sizeof(uint32_t));
  EXPECT_STREQ(a._schema[2].name, "alpha");
  EXPECT_EQ(a._schema[2].kind, detail::FieldKind::Scalar);
  EXPECT_EQ(a._schema[2].valueSize, sizeof(float));

  // Per-instance slot indices are identical across instances (registration
  // order is deterministic), but each vector is its own allocation.
  EXPECT_EQ(a._schema[0].slotIndex, b._schema[0].slotIndex);
  EXPECT_NE(static_cast<const void*>(a._schema.data()),
            static_cast<const void*>(b._schema.data()));
}

TEST(ShaderParamsSchema, KitchenSinkKinds) {
  KitchenSinkParams p;
  ASSERT_EQ(p._schema.size(), 5u);
  EXPECT_EQ(p._schema[0].kind, detail::FieldKind::UAVBuffer);
  EXPECT_EQ(p._schema[1].kind, detail::FieldKind::SRVBuffer);
  EXPECT_EQ(p._schema[2].kind, detail::FieldKind::SampledImage);
  EXPECT_EQ(p._schema[3].kind, detail::FieldKind::Sampler);
  EXPECT_EQ(p._schema[4].kind, detail::FieldKind::Scalar);
}

namespace {

// Build a synthetic ReflectionInfo so the test doesn't need DXC. Mimics
// what reflectSpirv would emit for a saxpy kernel:
//   set 0, binding 0  →  StorageBuffer "g_y"
//   push constant     →  offset=0, size=8 (uint count + float alpha)
ReflectionInfo makeSaxpyReflection() {
  ReflectionInfo ri;
  ri.stage = ShaderStage::Compute;
  ri.entryPoint = "main";

  DescriptorBindingInfo b;
  b.set = 0;
  b.binding = 0;
  b.type = DescriptorBindingInfo::Type::StorageBuffer;
  b.count = 1;
  b.name = "g_y";
  ri.bindings.push_back(b);

  PushConstantInfo pc;
  pc.offset = 0;
  pc.size = 8;
  ri.pushConstants.push_back(pc);
  return ri;
}

}  // namespace

TEST(ShaderParamsResolve, HappyPath) {
  SaxpyParams p;
  ReflectionInfo ri = makeSaxpyReflection();

  ASSERT_NO_THROW(p._resolve(ri));

  ASSERT_EQ(p._bindings.size(), 3u);

  // g_y → set 0 binding 0
  EXPECT_EQ(p._bindings[0].kind, detail::FieldKind::UAVBuffer);
  EXPECT_EQ(p._bindings[0].set, 0u);
  EXPECT_EQ(p._bindings[0].binding, 0u);

  // count → push constant at offset 0
  EXPECT_EQ(p._bindings[1].kind, detail::FieldKind::Scalar);
  EXPECT_EQ(p._bindings[1].pcOffset, 0u);
  EXPECT_EQ(p._bindings[1].pcSize, sizeof(uint32_t));

  // alpha → push constant at offset 4
  EXPECT_EQ(p._bindings[2].kind, detail::FieldKind::Scalar);
  EXPECT_EQ(p._bindings[2].pcOffset, 4u);
  EXPECT_EQ(p._bindings[2].pcSize, sizeof(float));
}

TEST(ShaderParamsResolve, MissingBindingThrows) {
  SaxpyParams p;
  ReflectionInfo ri = makeSaxpyReflection();
  // Strip the binding the params expects.
  ri.bindings.clear();

  EXPECT_THROW(p._resolve(ri), std::runtime_error);
}

TEST(ShaderParamsResolve, KindMismatchThrows) {
  SaxpyParams p;
  ReflectionInfo ri = makeSaxpyReflection();
  // Lie about g_y's type.
  ri.bindings[0].type = DescriptorBindingInfo::Type::UniformBuffer;

  // SaxpyParams declares g_y as UAV (= StorageBuffer). UAV vs UniformBuffer
  // is incompatible.
  EXPECT_THROW(p._resolve(ri), std::runtime_error);
}

TEST(ShaderParamsResolve, ScalarOverflowThrows) {
  SaxpyParams p;
  ReflectionInfo ri = makeSaxpyReflection();
  // Shrink push-constant block below what the SaxpyParams scalars need (8B).
  ri.pushConstants[0].size = 4;

  EXPECT_THROW(p._resolve(ri), std::runtime_error);
}

TEST(ShaderParamsResolve, EmptyPushConstantBlockOkIfNoScalar) {
  // Schema with only descriptor entries shouldn't care about absent PC block.
  NoScalarParams p;

  ReflectionInfo ri;
  ri.stage = ShaderStage::Compute;
  ri.entryPoint = "main";
  DescriptorBindingInfo b;
  b.set = 0;
  b.binding = 0;
  b.type = DescriptorBindingInfo::Type::StorageBuffer;
  b.count = 1;
  b.name = "g_y";
  ri.bindings.push_back(b);
  // Note: no push_constants entries.

  EXPECT_NO_THROW(p._resolve(ri));
  ASSERT_EQ(p._bindings.size(), 1u);
}

// Sanity: SHADER_PARAMS class can be defined inside a function — the schema
// is built in ctor, no static-init dependency. The previous (now reverted)
// per-field inline-static design forbade this.
TEST(ShaderParamsSchema, FunctionLocalDeclarationWorks) {
  SHADER_PARAMS_BEGIN(LocalParams)
    SHADER_PARAM_UAV   (BufferRef, g_x);
    SHADER_PARAM_SCALAR(uint32_t,  n);
  SHADER_PARAMS_END();

  LocalParams lp;
  ASSERT_EQ(lp._schema.size(), 2u);
  EXPECT_STREQ(lp._schema[0].name, "g_x");
  EXPECT_STREQ(lp._schema[1].name, "n");
  EXPECT_EQ(lp._schema[0].kind, detail::FieldKind::UAVBuffer);
  EXPECT_EQ(lp._schema[1].kind, detail::FieldKind::Scalar);
}
