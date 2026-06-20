// ============================================================================
// include/FluidSim/field-tags.h
// Well-known field name constants for the FluidContext registry.
// Use these instead of raw string literals to avoid typos.
// ============================================================================
#pragma once

#include <string_view>

namespace fluid::fields {

// ---- User-extensible fields (registry-based) ----
inline constexpr std::string_view kChemicalConcentration = "chemical_concentration";
inline constexpr std::string_view kVorticity = "vorticity";

// ---- Reserved names (for documentation; these map to Context members) ----
// These are NOT stored in the registry — they are direct members of the
// Context subclass. Listed here for documentation purposes only.
//
// "velocity"       → ctx.u / ctx.v / ctx.w
// "particles"      → ctx.positions / ctx.velocities
// "fluid_sdf"      → ctx.fluidSdf
// "collider_sdf"   → ctx.colliderSdf
// "u_valid"        → ctx.uValid
// "v_valid"        → ctx.vValid
// "w_valid"        → ctx.wValid
// "density"        → ctx.density
// "temperature"    → ctx.temperature

} // namespace fluid::fields
