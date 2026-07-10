//
// ipc-activation.h — shared (C++ + HLSL) candidate -> active-constraint mapping.
//
// Given a distance type and the candidate's vertex indices, produce the unified
// ConstraintPair (kind + up to 4 global vertex-block indices). This is the exact
// index reshuffle CPU does in IpcIntegrator::refreshActiveConstraintPairs /
// CollisionPair::appendConstraintPair, lifted to a single shared source so the
// GPU activation kernel emits byte-identical constraints.
//
#ifndef KISEKI_FEM_INCLUDE_FEM_SHARED_IPC_ACTIVATION_H_
#define KISEKI_FEM_INCLUDE_FEM_SHARED_IPC_ACTIVATION_H_

#include <fem/shared/ipc-distance-type.h>

// ConstraintKind codes (match enum class ConstraintKind in fem/ipc/constraint.h)
#define SH_CK_PP   0
#define SH_CK_PE   1
#define SH_CK_PT   2
#define SH_CK_EE   3
#define SH_CK_NONE 4  // inactive / unclassified — never emitted

SH_NS_BEGIN

// Unified active constraint: kind + 4 vertex-block indices (unused slots = -1),
// matching fem::ipc::ConstraintPair { ConstraintKind type; int indices[4]; }.
struct ShConstraintPair {
    int kind;
    int indices[4];
};

SH_INLINE ShConstraintPair shMakeConstraintPair() {
    ShConstraintPair cp;
    cp.kind = SH_CK_NONE;
    cp.indices[0] = -1;
    cp.indices[1] = -1;
    cp.indices[2] = -1;
    cp.indices[3] = -1;
    return cp;
}

// Vertex-Triangle candidate {v, t0, t1, t2} -> constraint (assumes active).
SH_INLINE ShConstraintPair shActivateVT(int type, int v, int t0, int t1, int t2) {
    ShConstraintPair cp = shMakeConstraintPair();
    if (type == SH_PT_P_A) {
        cp.kind = SH_CK_PP; cp.indices[0] = v; cp.indices[1] = t0;
    } else if (type == SH_PT_P_B) {
        cp.kind = SH_CK_PP; cp.indices[0] = v; cp.indices[1] = t1;
    } else if (type == SH_PT_P_C) {
        cp.kind = SH_CK_PP; cp.indices[0] = v; cp.indices[1] = t2;
    } else if (type == SH_PT_P_AB) {
        cp.kind = SH_CK_PE; cp.indices[0] = v; cp.indices[1] = t0; cp.indices[2] = t1;
    } else if (type == SH_PT_P_BC) {
        cp.kind = SH_CK_PE; cp.indices[0] = v; cp.indices[1] = t1; cp.indices[2] = t2;
    } else if (type == SH_PT_P_CA) {
        cp.kind = SH_CK_PE; cp.indices[0] = v; cp.indices[1] = t2; cp.indices[2] = t0;
    } else if (type == SH_PT_P_ABC) {
        cp.kind = SH_CK_PT; cp.indices[0] = v;
        cp.indices[1] = t0; cp.indices[2] = t1; cp.indices[3] = t2;
    }
    return cp;
}

// Edge-Edge candidate {a0, a1, b0, b1} -> constraint (assumes active).
SH_INLINE ShConstraintPair shActivateEE(int type, int a0, int a1, int b0, int b1) {
    ShConstraintPair cp = shMakeConstraintPair();
    if (type == SH_EE_A_C) {
        cp.kind = SH_CK_PP; cp.indices[0] = a0; cp.indices[1] = b0;
    } else if (type == SH_EE_A_D) {
        cp.kind = SH_CK_PP; cp.indices[0] = a0; cp.indices[1] = b1;
    } else if (type == SH_EE_B_C) {
        cp.kind = SH_CK_PP; cp.indices[0] = a1; cp.indices[1] = b0;
    } else if (type == SH_EE_B_D) {
        cp.kind = SH_CK_PP; cp.indices[0] = a1; cp.indices[1] = b1;
    } else if (type == SH_EE_AB_C) {
        cp.kind = SH_CK_PE; cp.indices[0] = b0; cp.indices[1] = a0; cp.indices[2] = a1;
    } else if (type == SH_EE_AB_D) {
        cp.kind = SH_CK_PE; cp.indices[0] = b1; cp.indices[1] = a0; cp.indices[2] = a1;
    } else if (type == SH_EE_A_CD) {
        cp.kind = SH_CK_PE; cp.indices[0] = a0; cp.indices[1] = b0; cp.indices[2] = b1;
    } else if (type == SH_EE_B_CD) {
        cp.kind = SH_CK_PE; cp.indices[0] = a1; cp.indices[1] = b0; cp.indices[2] = b1;
    } else if (type == SH_EE_AB_CD) {
        cp.kind = SH_CK_EE;
        cp.indices[0] = a0; cp.indices[1] = a1; cp.indices[2] = b0; cp.indices[3] = b1;
    }
    return cp;
}

SH_NS_END

#endif  // KISEKI_FEM_INCLUDE_FEM_SHARED_IPC_ACTIVATION_H_
