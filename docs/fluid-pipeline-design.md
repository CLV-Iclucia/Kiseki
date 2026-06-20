# FluidSim 可组合管线架构设计文档

## 求解器组合 + Context 注册表 + 配置解耦

**版本**: v2.0  
**日期**: 2026-06-19  
**作者**: creeper  
**状态**: Draft  
**前置文档**: `docs/fluid-gpu-design.md`（v3.1，CPU/GPU 后端设计）

---

## 目录

1. [动机与目标](#1-动机与目标)
2. [核心设计理念](#2-核心设计理念)
3. [架构总览](#3-架构总览)
4. [FluidContext 详细设计](#4-fluidcontext-详细设计)
5. [FluidModularSolver 接口](#5-fluidmodularsolver-接口)
6. [FluidSimulator 编排器](#6-fluidsimulator-编排器)
7. [Backend 与 Solver 的类型安全](#7-backend-与-solver-的类型安全)
8. [Solver 间数据传递策略](#8-solver-间数据传递策略)
9. [配置系统设计](#9-配置系统设计)
10. [预设工厂（Presets）](#10-预设工厂presets)
11. [文件结构](#11-文件结构)
12. [与现有代码的关系](#12-与现有代码的关系)
13. [迁移路径](#13-迁移路径)
14. [附录](#附录)

---

## 1. 动机与目标

### 1.1 问题陈述

流体模拟涵盖多种类型（自由液面、烟雾、云体、多相流等），这些类型：
- **不能统一为单一 Pipeline 类**：输入/输出数据结构不同（粒子 vs 密度场 vs 混合）
- **有大量可复用算法模块**：压力投影、平流、CG solver 等跨类型共享
- **每种类型 × 每种后端 = 类型组合爆炸**：如果把 Pipeline 作为独立类，则需要 `CPUFreeSurface`、`GPUFreeSurface`、`CPUSmoke`、`GPUSmoke`...

### 1.2 核心洞察

> **Pipeline 不是类型，而是组合。**

不同的管线来自不同的求解器组合。如果允许定制求解器、允许自由组合，就能避免 Pipeline + Backend 的类型爆炸。对于常见的固定组合（如自由液面 FLIP），再提供专门的预设工厂即可。

### 1.3 设计目标

| 目标 | 说明 |
|------|------|
| **无类型爆炸** | 新增物理类型 = 新增 Solver，不需要新的 Pipeline 类 |
| **可组合** | 用户可以自由搭配 Solver 构成任意管线 |
| **自定义物理量** | 支持注册任意命名的数据，无需修改 Context 头文件 |
| **配置解耦** | 每个 Solver 自带配置，无全局 God Config |
| **后端隔离** | 编译期区分后端，具体 Solver 直接拿到强类型 Context |
| **渐进式开发** | 当前只实现 FreeSurface × {CPU, GPU}，以后无痛扩展 |
| **Python 友好** | 简洁的公共接口，易于绑定 |
| **构造即就绪** | 构造完成后管线不可变，无 two-phase init |

---

## 2. 核心设计理念

### 2.1 三层数据归属

| 归属 | 判断标准 | 访问方式 | 生命周期 |
|------|---------|---------|---------|
| **Context 固定成员** | 任何欧拉流体模拟都必然存在 | `ctx.velocity()` | 整个模拟 |
| **Context 注册表** | 全局共享但并非所有模拟都有 | `ctx.get<T>("name")` | 整个模拟 |
| **Solver 自有** | 局部、短生命周期、仅自己或相邻 Solver 关心 | Solver 内部成员 | 一个时间步内 |

### 2.2 黑板模式 (Blackboard Pattern)

Context 是所有 Solver 的「公共黑板」。Solver 之间**不直接传数据**，全部通过 Context 的固定成员或注册表间接通信。

### 2.3 组合优于继承

不同 Pipeline 不是不同的 class，而是同一个 `FluidSimulator` 容器里塞了不同的 Solver 组合。

### 2.4 构造即就绪

FluidSimulator 构造时接收完整的 Context + Solver 列表，构造完成后管线不可变。没有 `addSolver()`、没有 `initialize()`，不存在"忘了初始化"的可能。

---

## 3. 架构总览

```
┌───────────────────────────────────────────────────────────────────┐
│                      FluidSimulator (编排器)                        │
│   构造时接收 FluidContext + vector<FluidModularSolver>              │
│   构造后不可变，只有 step() 驱动                                     │
└───────────┬───────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────────┐
│                      FluidContext (公共数据)                        │
│                                                                    │
│   固定成员:                                                         │
│     domain, gridSpacing, velocity (MAC), colliderSdf               │
│                                                                    │
│   注册表 (std::any map):                                            │
│     "particles" → CPUParticleData                                  │
│     "fluid_sdf" → CPUSdfField                                      │
│     "density"   → CPUScalarField       (烟雾)                      │
│     "u_valid"   → CPUFlagField         (外推用)                    │
│     ...                                                            │
└───────────────────────────────────────────────────────────────────┘
            ▲           ▲           ▲           ▲
            │           │           │           │
     ┌──────┘    ┌──────┘    ┌──────┘    ┌──────┘
     │           │           │           │
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│Advector │ │Reconstr.│ │  Force  │ │Projector│
│         │ │         │ │         │ │         │
│config:  │ │config:  │ │config:  │ │config:  │
│AdvCfg  │ │RecCfg   │ │ForceCfg │ │ProjCfg  │
└─────────┘ └─────────┘ └─────────┘ └─────────┘
   各自持有内部 buffer、工作数据
```

### 3.1 对比现有架构

| 现有 (fluid-gpu-design.md v3.1) | 新设计 |
|------|--------|
| `FluidBackend` = 整条管线的 God Class | `FluidSimulator` = 薄编排器，逻辑在 Solver 中 |
| CPU/GPU 各一个大的 Backend 类 | Context 子类区分后端，Solver 组件化 |
| 接口 `initialize/step/readback/update*` 5 个虚方法 | `FluidModularSolver::solve(ctx, dt)` 单方法 |
| 新增物理类型需要改 Backend 接口 | 新增物理类型 = 新增 Solver + 注册字段 |
| 构造后需要调 `initialize()` | 构造即就绪 |

---

## 4. FluidContext 详细设计

### 4.1 设计原则

- **固定成员**：速度场、碰撞体 SDF、域信息——任何基于网格的流体模拟都有这些，直接作为 Context 子类的成员
- **注册表**：粒子、fluid SDF、密度场、温度场、validity flags、自定义物理量——使用 `std::any` 按名存储
- **后端多态**：CPU Context 和 GPU Context 是子类，各自持有具体的数据结构

### 4.2 接口定义

```cpp
// ============================================================================
// include/FluidSim/fluid-context.h
// ============================================================================
#pragma once

#include <FluidSim/fluid-types.h>
#include <Core/core.h>
#include <any>
#include <string>
#include <string_view>
#include <unordered_map>
#include <stdexcept>
#include <vector>

namespace fluid {

class FluidContext : NonCopyable {
public:
    virtual ~FluidContext() = default;

    // ============ 固定成员（所有欧拉流体都有）============

    FluidDomain domain{};
    Real        gridSpacing{};

    // 速度场和碰撞体 SDF 由子类提供具体类型的访问接口

    // ============ 注册表（std::any）============

    /// 注册数据（已存在则覆盖）
    template<typename T>
    T& set(std::string_view name, T value);

    /// 注册数据（已存在则返回已有的，不存在则原地构造）
    template<typename T, typename... Args>
    T& ensure(std::string_view name, Args&&... args);

    /// 获取数据（不存在或类型不匹配则抛异常）
    template<typename T>
    T& get(std::string_view name);

    template<typename T>
    const T& get(std::string_view name) const;

    /// 尝试获取（不存在或类型不匹配返回 nullptr）
    template<typename T>
    T* tryGet(std::string_view name);

    template<typename T>
    const T* tryGet(std::string_view name) const;

    /// 查询是否存在
    bool has(std::string_view name) const;

    /// 移除
    void remove(std::string_view name);

    /// 获取所有已注册字段的名称
    std::vector<std::string> keys() const;

    // ============ 帧钩子 ============

    /// GPU 端用于管理 CommandList，CPU 端空实现
    virtual void beginFrame() {}
    virtual void endFrame() {}

private:
    std::unordered_map<std::string, std::any> registry_;
};

// ============================================================================
// 模板实现
// ============================================================================

template<typename T>
T& FluidContext::set(std::string_view name, T value) {
    std::string key(name);
    registry_[key] = std::move(value);
    return std::any_cast<T&>(registry_[key]);
}

template<typename T, typename... Args>
T& FluidContext::ensure(std::string_view name, Args&&... args) {
    std::string key(name);
    auto it = registry_.find(key);
    if (it != registry_.end()) {
        if (auto* ptr = std::any_cast<T>(&it->second))
            return *ptr;
        throw std::runtime_error(
            "Registry key '" + key + "' exists but with incompatible type");
    }
    registry_[key] = T(std::forward<Args>(args)...);
    return std::any_cast<T&>(registry_[key]);
}

template<typename T>
T& FluidContext::get(std::string_view name) {
    std::string key(name);
    auto it = registry_.find(key);
    if (it == registry_.end())
        throw std::runtime_error("Registry key '" + key + "' not found");
    auto* ptr = std::any_cast<T>(&it->second);
    if (!ptr)
        throw std::runtime_error("Registry key '" + key + "' type mismatch");
    return *ptr;
}

template<typename T>
const T& FluidContext::get(std::string_view name) const {
    return const_cast<FluidContext*>(this)->get<T>(name);
}

template<typename T>
T* FluidContext::tryGet(std::string_view name) {
    std::string key(name);
    auto it = registry_.find(key);
    if (it == registry_.end()) return nullptr;
    return std::any_cast<T>(&it->second);
}

template<typename T>
const T* FluidContext::tryGet(std::string_view name) const {
    return const_cast<FluidContext*>(this)->tryGet<T>(name);
}

} // namespace fluid
```

### 4.3 Context 子类的数据成员策略

**原则**：注册表只用于真正无法预知的自定义物理量。对于各后端已知的常见数据（particles、fluid SDF、validity flags 等），直接作为 Context 子类的成员。

| 数据 | 归属 | 理由 |
|------|------|------|
| velocity (MAC) | Context 子类成员 | 所有欧拉模拟都有 |
| collider SDF | Context 子类成员 | 所有模拟都有 |
| particles | Context 子类成员 | 自由液面模拟必有，构造时即确定 |
| fluid SDF | Context 子类成员 | 自由液面模拟必有 |
| validity flags (u/v/w) | Context 子类成员 | 外推阶段通用 |
| density / temperature | Context 子类成员 | 烟雾模拟必有，构造时即确定 |
| 用户自定义物理量 | 注册表 (`std::any`) | 无法预知 |

注册表 (`std::any` map) 退化为**只服务于用户扩展**的极少数场景。大部分常用数据直接通过 `ctx.particles`、`ctx.fluidSdf` 等成员访问。

**注意**：不是所有成员在所有模拟中都会被用到。对于可选的成员（如 particles 在烟雾模拟中不需要），使用 `std::optional` 或指针表达可选性。

### 4.4 CPU Context

```cpp
// ============================================================================
// include/FluidSim/cpu/cpu-context.h
// ============================================================================
#pragma once

#include <FluidSim/fluid-context.h>
#include <FluidSim/cpu/sdf.h>
#include <Spatify/grids.h>
#include <Spatify/arrays.h>

namespace fluid::cpu {

// ---- CPU Context ----

class CPUFluidContext : public FluidContext {
public:
    explicit CPUFluidContext(const FluidDomain& domain);

    using FaceGridU = spatify::FaceCentredGrid<Real, Real, 3, 0>;
    using FaceGridV = spatify::FaceCentredGrid<Real, Real, 3, 1>;
    using FaceGridW = spatify::FaceCentredGrid<Real, Real, 3, 2>;

    // ============ 所有模拟都有 ============

    std::unique_ptr<FaceGridU> u, uBuf;
    std::unique_ptr<FaceGridV> v, vBuf;
    std::unique_ptr<FaceGridW> w, wBuf;

    std::unique_ptr<SDF<3>> colliderSdf;

    std::unique_ptr<spatify::Array3D<char>> uValid, vValid, wValid;
    std::unique_ptr<spatify::Array3D<char>> uValidBuf, vValidBuf, wValidBuf;

    // ============ 自由液面专用（可选）============

    std::vector<Vec3d> positions;
    std::vector<Vec3d> velocities;

    std::unique_ptr<SDF<3>> fluidSdf;
    std::unique_ptr<SDF<3>> fluidSdfBuf;
    std::unique_ptr<spatify::Array3D<char>> sdfValid, sdfValidBuf;

    spatify::Array3D<Real> uw, vw, ww;


    std::unique_ptr<spatify::Array3D<Real>> density;
    std::unique_ptr<spatify::Array3D<Real>> temperature;

    FaceGridU& ug() { return *u; }
    FaceGridV& vg() { return *v; }
    FaceGridW& wg() { return *w; }
};

} // namespace fluid::cpu
```

### 4.5 GPU Context

```cpp
// ============================================================================
// include/FluidSim/gpu/gpu-context.h
// ============================================================================
#pragma once

#include <FluidSim/fluid-context.h>
#include <RHI/rhi.h>

namespace fluid::gpu {

class GPUFluidContext : public FluidContext {
public:
    explicit GPUFluidContext(sim::rhi::Device& device, const FluidDomain& domain,
                            uint32_t numParticles);

    // ============ 所有模拟都有 ============

    // 速度场 (SSBO)
    sim::rhi::BufferRef uGrid, vGrid, wGrid;

    // 有效性标记 (SSBO)
    sim::rhi::BufferRef uValid, vValid, wValid;

    // 碰撞体 SDF (Image D3)
    sim::rhi::ImageRef  colliderSdfImg;
    sim::rhi::SamplerRef sdfSampler;

    // 维度
    Vec3i gridSize{};

    // ============ 自由液面专用 ============

    // 粒子 (SSBO, struct { double3 pos; double3 vel; })
    sim::rhi::BufferRef particles;
    uint32_t numParticles{};

    // 流体 SDF (Image D3)
    sim::rhi::ImageRef fluidSdfImg;

    // ---- CommandList 管理 ----
    void beginFrame() override;
    void endFrame() override;

    sim::rhi::Device& device() { return device_; }
    sim::rhi::CommandList& cmd() { return *activeCmd_; }

private:
    sim::rhi::Device& device_;
    sim::rhi::CommandList* activeCmd_{};
};

} // namespace fluid::gpu
```

---

## 5. FluidModularSolver 接口

### 5.1 基类 + 后端中间层

```cpp
// ============================================================================
// include/FluidSim/fluid-solver.h
// ============================================================================
#pragma once

#include <FluidSim/fluid-context.h>
#include <Core/properties.h>
#include <string>
#include <string_view>

namespace fluid {

// ---- 通用基类（Python binding 和 Simulator 持有的类型）----

class FluidModularSolver : NonCopyable {
public:
    explicit FluidModularSolver(std::string name) : name_(std::move(name)) {}
    virtual ~FluidModularSolver() = default;

    /// 执行一个子步
    virtual void solve(FluidContext& ctx, Real dt) = 0;

    /// 运行时配置更新（类型擦除，Python/UI 友好）
    virtual void configure(const core::Properties& props) {}

    /// 查询当前配置
    virtual core::Properties currentConfig() const { return {}; }

    /// Solver 名称（调试/查找用）
    std::string_view name() const { return name_; }

private:
    std::string name_;
};

} // namespace fluid
```

### 5.2 后端中间层基类

后端中间层负责将 `FluidContext&` 转换为具体的 Context 子类，消除每个 Solver 中的重复 cast：

```cpp
// ============================================================================
// include/FluidSim/cpu/cpu-solver.h
// ============================================================================
#pragma once

#include <FluidSim/fluid-solver.h>
#include <FluidSim/cpu/cpu-context.h>

namespace fluid::cpu {

class CPUModularSolver : public FluidModularSolver {
public:
    using FluidModularSolver::FluidModularSolver;  // 继承构造

    void solve(FluidContext& ctx, Real dt) final {
        solve(static_cast<CPUFluidContext&>(ctx), dt);
    }

    /// CPU Solver 实现这个——直接拿到强类型 Context
    virtual void solve(CPUFluidContext& ctx, Real dt) = 0;
};

} // namespace fluid::cpu
```

```cpp
// ============================================================================
// include/FluidSim/gpu/gpu-solver.h
// ============================================================================
#pragma once

#include <FluidSim/fluid-solver.h>
#include <FluidSim/gpu/gpu-context.h>

namespace fluid::gpu {

class GPUModularSolver : public FluidModularSolver {
public:
    using FluidModularSolver::FluidModularSolver;  // 继承构造

    void solve(FluidContext& ctx, Real dt) final {
        solve(static_cast<GPUFluidContext&>(ctx), dt);
    }

    /// GPU Solver 实现这个——直接拿到强类型 Context
    virtual void solve(GPUFluidContext& ctx, Real dt) = 0;
};

} // namespace fluid::gpu
```

### 5.3 设计说明

| 设计决策 | 说明 |
|---------|------|
| `name` 是成员，非虚方法 | 构造时传入，无多态必要 |
| `static_cast` 只出现一次 | 在中间层基类的 `solve(FluidContext&)` 中，具体 Solver 不关心 |
| 无 `declare()` 阶段 | Solver 在构造时即可操作 Context 注册表（Context 先于 Solver 构造） |
| 管线不可变 | 无 `addSolver`/`removeSolver`，构造后定死 |

---

## 6. FluidSimulator 编排器

### 6.1 接口

```cpp
// ============================================================================
// include/FluidSim/fluid-simulator.h
// ============================================================================
#pragma once

#include <FluidSim/fluid-context.h>
#include <FluidSim/fluid-solver.h>
#include <Core/animation.h>
#include <memory>
#include <vector>

namespace fluid {

class FluidSimulator : public core::Animation {
public:
    /// 构造即就绪：传入 Context 和完整的 Solver 列表
    FluidSimulator(std::unique_ptr<FluidContext> ctx,
                   std::vector<std::shared_ptr<FluidModularSolver>> solvers);

    // ---- 运行 ----
    void step(core::Frame& frame) override;

    // ---- 访问 ----
    FluidContext& context() { return *ctx_; }
    const FluidContext& context() const { return *ctx_; }

    /// 按名称查找 Solver（用于运行时配置更新）
    FluidModularSolver* findSolver(std::string_view name);

    template<typename T>
    T* findSolver(std::string_view name) {
        return dynamic_cast<T*>(findSolver(name));
    }

    const std::vector<std::shared_ptr<FluidModularSolver>>& solvers() const {
        return solvers_;
    }

private:
    std::unique_ptr<FluidContext> ctx_;
    std::vector<std::shared_ptr<FluidModularSolver>> solvers_;

    Real computeCFL() const;
};

} // namespace fluid
```

### 6.2 实现

```cpp
FluidSimulator::FluidSimulator(
    std::unique_ptr<FluidContext> ctx,
    std::vector<std::shared_ptr<FluidModularSolver>> solvers)
    : ctx_(std::move(ctx))
    , solvers_(std::move(solvers))
{
    // 构造完成即就绪，无额外初始化步骤
}

void FluidSimulator::step(core::Frame& frame) {
    Real dt = frame.dt;
    Real t = 0;

    ctx_->beginFrame();  // GPU: begin command list; CPU: no-op

    while (t < dt) {
        Real subDt = std::min(computeCFL(), dt - t);
        for (auto& solver : solvers_) {
            solver->solve(*ctx_, subDt);
        }
        t += subDt;
    }

    ctx_->endFrame();    // GPU: submit + wait; CPU: no-op
}

FluidModularSolver* FluidSimulator::findSolver(std::string_view name) {
    for (auto& s : solvers_)
        if (s->name() == name) return s.get();
    return nullptr;
}
```

---

## 7. Backend 与 Solver 的类型安全

### 7.1 核心原则

> **具体 Solver 继承后端中间层基类（`CPUModularSolver` / `GPUModularSolver`），直接拿到强类型 Context。cast 只在中间层一处发生。**
> **同一个 FluidSimulator 实例内所有 Solver 属于同一后端，由工厂保证。**

### 7.2 类层次

```
FluidModularSolver              ← Simulator 持有的基类
├── CPUModularSolver            ← CPU 中间层（final solve(FluidContext&) → solve(CPUFluidContext&)）
│   ├── cpu::CPUAdvectorP2G     ← 具体实现
│   ├── cpu::CPUExtrapolator
│   ├── cpu::CPUReconstructor
│   ├── cpu::CPUForceApplicator
│   ├── cpu::CPUProjector
│   └── cpu::CPUAdvectorG2P
└── GPUModularSolver            ← GPU 中间层
    ├── gpu::GPUAdvectorP2G
    ├── gpu::GPUExtrapolator
    ├── gpu::GPUProjector
    └── ...
```

### 7.3 CPU Solver 实现示例

```cpp
namespace fluid::cpu {

class CPUProjector : public CPUModularSolver {
public:
    struct Config {
        PreconditionerMethod preconditioner = PreconditionerMethod::ModifiedIncompleteCholesky;
        int maxIters = 300;
        Real tolerance = 1e-4;
        Real density = 1000.0;
    };

    CPUProjector(CPUFluidContext& ctx, Config config)
        : CPUModularSolver("cpu_projector"), config_(config)
    {
        // 构造时即可根据 ctx.domain 分配内部 buffer
        auto [nx, ny, nz] = ctx.domain.resolution;
        Adiag_.resize(nx, ny, nz);
        rhs_.resize(nx, ny, nz);
        pressure_.resize(nx, ny, nz);
        // ...
    }

    void solve(CPUFluidContext& ctx, Real dt) override {
        // 直接访问 Context 成员，无任何 cast 或 hash lookup
        buildSystem(*ctx.u, *ctx.v, *ctx.w, *ctx.fluidSdf, *ctx.colliderSdf, dt);
        solvePressure();
        project(*ctx.u, *ctx.v, *ctx.w, *ctx.fluidSdf, dt);
    }

    void configure(const core::Properties& props) override {
        if (auto v = props.tryGet<int>("max_iters"))
            config_.maxIters = *v;
        if (auto v = props.tryGet<Real>("tolerance"))
            config_.tolerance = *v;
    }

private:
    Config config_;

    // 内部数据——不暴露到 Context
    spatify::Array3D<Real> Adiag_;
    std::array<spatify::Array3D<Real>, 6> Aneighbour_;
    spatify::Array3D<Real> rhs_;
    spatify::Array3D<Real> pressure_;
    std::unique_ptr<CgSolver3D> cgSolver_;
};

} // namespace fluid::cpu
```

### 7.4 GPU Solver 实现示例

```cpp
namespace fluid::gpu {

class GPUProjector : public GPUModularSolver {
public:
    struct Config {
        PreconditionerMethod preconditioner = PreconditionerMethod::Jacobi;
        int maxIters = 200;
        Real density = 1000.0;
    };

    GPUProjector(GPUFluidContext& ctx, Config config)
        : GPUModularSolver("gpu_projector"), config_(config)
    {
        auto& dev = ctx.device();
        auto [nx, ny, nz] = ctx.gridSize;
        uint32_t nc = nx * ny * nz;
        // 分配自有 buffer
        pressure_ = dev.createBuffer({sizeof(double)*nc, BufferUsage::Storage});
        Adiag_ = dev.createBuffer({sizeof(double)*nc, BufferUsage::Storage});
        // ... 其它 buffer 和 pipeline 创建
    }

    void solve(GPUFluidContext& ctx, Real dt) override {
        auto& cmd = ctx.cmd();
        buildWeightsAndSystem(cmd, ctx, dt);
        cgSolve(cmd);
        projectVelocities(cmd, ctx, dt);
    }

private:
    Config config_;
    sim::rhi::BufferRef pressure_, Adiag_;
    // ...
};

} // namespace fluid::gpu
```

---

## 8. Solver 间数据传递策略

### 8.1 全局数据 → Context 成员直接访问

```cpp
// Advector 写入 velocity → Projector 读取 velocity
// 两者都通过 ctx.u / ctx.v / ctx.w 直接访问
void CPUAdvectorP2G::solve(CPUFluidContext& ctx, Real dt) {
    // particles 也是 Context 成员
    // P2G scatter: 读 ctx.positions/velocities → 写 ctx.u, ctx.v, ctx.w, ctx.uValid...
}

void CPUProjector::solve(CPUFluidContext& ctx, Real dt) {
    // 直接读取 ctx.u, ctx.v, ctx.w → 投影修正 → 写回
}
```

### 8.2 Solver 内部数据 → 自己持有

```cpp
class CPUProjector : public CPUModularSolver {
    // Adiag, Aneighbour, rhs, CG 中间向量——只有 Projector 自己用
    spatify::Array3D<Real> Adiag_;
    spatify::Array3D<Real> rhs_;
    // ...
};
```

### 8.3 相邻 Solver 间的共享数据 → Context 成员

**场景**：Advector 做完 P2G 后产出 `validity flags`，Extrapolator 需要读取。

**解决方案**：validity flags 是 `CPUFluidContext` 的成员（`ctx.uValid` 等），Advector 写入、Extrapolator 读取，两者互不知道对方的存在：

```cpp
class CPUAdvectorP2G : public CPUModularSolver {
public:
    CPUAdvectorP2G(CPUFluidContext& ctx, Config config)
        : CPUModularSolver("cpu_advector_p2g"), config_(config) {}

    void solve(CPUFluidContext& ctx, Real dt) override {
        // P2G 时同时写入 velocity 和 validity flags
        // 直接访问 ctx 成员，无需注册
        ctx.uValid->fill(0);
        ctx.vValid->fill(0);
        ctx.wValid->fill(0);
        // ... scatter particles to grid ...
    }
};

class CPUExtrapolator : public CPUModularSolver {
public:
    CPUExtrapolator(CPUFluidContext& ctx)
        : CPUModularSolver("cpu_extrapolator") {}

    void solve(CPUFluidContext& ctx, Real dt) override {
        // 直接读取 ctx 成员
        extrapolate(*ctx.u, *ctx.uBuf, *ctx.uValid, *ctx.uValidBuf, 10);
        extrapolate(*ctx.v, *ctx.vBuf, *ctx.vValid, *ctx.vValidBuf, 10);
        extrapolate(*ctx.w, *ctx.wBuf, *ctx.wValid, *ctx.wValidBuf, 10);
    }
};
```

### 8.4 注册表的使用场景

注册表 (`std::any` map) **仅用于用户自定义的未知物理量**：

```cpp
class ChemicalDiffusionSolver : public CPUModularSolver {
public:
    ChemicalDiffusionSolver(CPUFluidContext& ctx, Config config)
        : CPUModularSolver("chemical_diffusion"), config_(config)
    {
        // 自定义物理量——Context 不可能预知，所以走注册表
        auto [nx, ny, nz] = ctx.domain.resolution;
        ctx.ensure<spatify::Array3D<Real>>("chemical_concentration", nx, ny, nz);
    }

    void solve(CPUFluidContext& ctx, Real dt) override {
        auto& conc = ctx.get<spatify::Array3D<Real>>("chemical_concentration");
        // 扩散方程求解...
    }
};
```

### 8.5 决策流程

```
数据 X 被 Solver A 产出:

  X 是已知的常见流体模拟数据？
  (velocity, particles, SDF, validity flags, density, temperature...)
    ├── 是 → Context 子类的成员（直接访问）
    └── 否 → X 只有 Solver A 自己用？
              ├── 是 → A 的私有成员
              └── 否 → Context 注册表（std::any map，仅用户扩展场景）
```

---

## 9. 配置系统设计

### 9.1 原则

> **配置跟着 Solver 走，不集中管理。**

每个 Solver 有自己的强类型 Config struct，在构造时传入。运行时可通过 `configure(Properties)` 更新。

### 9.2 强类型配置（C++ 端）

```cpp
struct AdvectorConfig {
    AdvectionMethod method = AdvectionMethod::FLIP;
    Real flipBlend = 0.97;
    int rk3Steps = 3;
};

struct ProjectorConfig {
    PreconditionerMethod preconditioner = PreconditionerMethod::ModifiedIncompleteCholesky;
    int maxIters = 300;
    Real tolerance = 1e-4;
    Real density = 1000.0;
};

struct ReconstructorConfig {
    ReconstructorMethod method = ReconstructorMethod::Naive;
    Real particleRadius = 1.5;  // 以 dx 为单位
    int smoothIters = 3;
};

struct ForceConfig {
    Vec3d gravity{0.0, -9.8, 0.0};
};

struct BuoyancyConfig {
    Real alpha = 0.001;
    Real beta = 0.003;
    Real ambientTemp = 300.0;
};
```

### 9.3 类型擦除配置（Python/UI 端）

```cpp
// Python 调用示例:
//   sim.find_solver("cpu_advector_p2g").configure({"flip_blend": 0.95})

void CPUAdvectorP2G::configure(const core::Properties& props) {
    if (auto v = props.tryGet<Real>("flip_blend"))
        config_.flipBlend = *v;
    if (auto v = props.tryGet<std::string>("method")) {
        if (*v == "PIC") config_.method = AdvectionMethod::PIC;
        else if (*v == "FLIP") config_.method = AdvectionMethod::FLIP;
        else if (*v == "APIC") config_.method = AdvectionMethod::APIC;
    }
}
```

### 9.4 配置层次总结

| 层次 | 时机 | 接口 | 用途 |
|------|------|------|------|
| 构造时 Config struct | 创建 Solver 时 | `CPUProjector(ctx, Config{...})` | 初始配置，强类型 |
| 运行时 Properties | Python/UI 调参 | `solver->configure(props)` | 动态更新，类型擦除 |
| 预设工厂 | 一站式创建 | `presets::createFreeSurface(cfg)` | 把散乱的 Config 打包 |

---

## 10. 预设工厂（Presets）

### 10.1 设计

```cpp
// ============================================================================
// include/FluidSim/presets.h
// ============================================================================
#pragma once

#include <FluidSim/fluid-simulator.h>
#include <memory>

namespace fluid {

enum class BackendType { CPU, GPU };

struct FreeSurfaceConfig {
    FluidDomain domain;
    InitialFluid initialFluid;
    std::optional<Mesh> colliderMesh;

    AdvectorConfig advector;
    ProjectorConfig projector;
    ReconstructorConfig reconstructor;
    ForceConfig force;

    Real maxCfl = 5.0;
};

struct SmokeConfig {
    FluidDomain domain;
    // 初始密度/温度分布
    // ...

    ProjectorConfig projector;
    ForceConfig force;
    BuoyancyConfig buoyancy;

    Real maxCfl = 5.0;
};

namespace presets {

std::unique_ptr<FluidSimulator> createFreeSurface(
    const FreeSurfaceConfig& config,
    BackendType backend = BackendType::CPU);

std::unique_ptr<FluidSimulator> createSmoke(
    const SmokeConfig& config,
    BackendType backend = BackendType::CPU);

} // namespace presets
} // namespace fluid
```

### 10.2 实现示例

```cpp
std::unique_ptr<FluidSimulator> presets::createFreeSurface(
    const FreeSurfaceConfig& config, BackendType backend)
{
    if (backend == BackendType::CPU) {
        // 1. 创建 Context（构造时分配所有网格/SDF 等成员）
        auto ctx = std::make_unique<cpu::CPUFluidContext>(config.domain);

        // 2. 填充初始数据（直接写成员）
        ctx->positions = config.initialFluid.positions;
        ctx->velocities = config.initialFluid.velocities;

        // 3. 创建 Solver（构造时传入 ctx 引用 + 各自 config）
        auto advP2G = std::make_shared<cpu::CPUAdvectorP2G>(*ctx, config.advector);
        auto extrap = std::make_shared<cpu::CPUExtrapolator>(*ctx);
        auto recon  = std::make_shared<cpu::CPUReconstructor>(*ctx, config.reconstructor);
        auto force  = std::make_shared<cpu::CPUForceApplicator>(*ctx, config.force);
        auto proj   = std::make_shared<cpu::CPUProjector>(*ctx, config.projector);
        auto advG2P = std::make_shared<cpu::CPUAdvectorG2P>(*ctx, config.advector);

        // 4. 一次构造 Simulator（管线从此不可变）
        return std::make_unique<FluidSimulator>(
            std::move(ctx),
            std::vector<std::shared_ptr<FluidModularSolver>>{
                advP2G, extrap, recon, force, proj, advG2P
            }
        );
    }
    else {
        // GPU 版本类似
        // ...
    }
}
```

### 10.3 手动组装示例

```cpp
// 用户自定义管线
auto ctx = std::make_unique<cpu::CPUFluidContext>(myDomain);

// 填充粒子（直接写成员）
ctx->positions = myPositions;
ctx->velocities = myVelocities;

// 注册自定义物理量（仅用户扩展才走注册表）
ctx->ensure<spatify::Array3D<Real>>("chemical_concentration", nx, ny, nz);

// 创建 Solver
auto advP2G = std::make_shared<cpu::CPUAdvectorP2G>(*ctx, advCfg);
auto proj   = std::make_shared<cpu::CPUProjector>(*ctx, projCfg);
auto chem   = std::make_shared<MyChemicalDiffusionSolver>(*ctx, diffCfg);
auto advG2P = std::make_shared<cpu::CPUAdvectorG2P>(*ctx, advCfg);

// 一次构造
auto sim = std::make_unique<FluidSimulator>(
    std::move(ctx),
    std::vector<std::shared_ptr<FluidModularSolver>>{ advP2G, proj, chem, advG2P }
);
```

---

## 11. 文件结构

```
FluidSim/
├── include/FluidSim/
│   ├── fluid-context.h            ← [NEW] FluidContext 基类 + std::any 注册表
│   ├── fluid-solver.h             ← [NEW] FluidModularSolver 基类
│   ├── fluid-simulator.h          ← [REWRITE] 编排器（构造即就绪）
│   ├── field-tags.h               ← [NEW] Well-known field name constants
│   ├── presets.h                  ← [NEW] 预设工厂
│   ├── fluid-types.h              ← 保留（纯数据类型）
│   ├── kernel.h                   ← 保留
│   ├── cpu/
│   │   ├── cpu-context.h          ← [NEW] CPUFluidContext + CPU 数据类型
│   │   ├── cpu-solver.h           ← [NEW] CPUModularSolver 中间层
│   │   ├── cpu-advector.h         ← [NEW] CPUAdvectorP2G / CPUAdvectorG2P
│   │   ├── cpu-projector.h        ← [NEW] CPUProjector
│   │   ├── cpu-reconstructor.h    ← [NEW] CPUReconstructor
│   │   ├── cpu-extrapolator.h     ← [NEW] CPUExtrapolator
│   │   ├── cpu-force.h            ← [NEW] CPUForceApplicator
│   │   ├── advect-solver.h        ← 保留（内部实现）
│   │   ├── project-solver.h       ← 保留（内部实现）
│   │   ├── sdf.h                  ← 保留
│   │   ├── rebuild-surface.h      ← 保留
│   │   └── util.h                 ← 保留
│   └── gpu/
│       ├── gpu-context.h           ← [NEW] GPUFluidContext
│       ├── gpu-solver.h            ← [NEW] GPUModularSolver 中间层
│       ├── gpu-advector.h          ← [KEEP] 适配为 GPUModularSolver 子类
│       ├── gpu-projector.h         ← [KEEP] 适配为 GPUModularSolver 子类
│       ├── gpu-reconstructor.h     ← [KEEP] 适配
│       ├── gpu-extrapolator.h      ← [NEW]
│       ├── gpu-force.h             ← [NEW]
│       ├── gpu-pcg-solver.h        ← 保留（GPUProjector 内部使用）
│       └── gpu-jacobi-solver.h     ← 保留（GPUProjector 内部使用）
├── src/
│   ├── fluid-simulator.cc         ← [REWRITE]
│   ├── fluid-context.cc           ← [NEW] has/remove/keys
│   ├── presets.cc                  ← [NEW]
│   ├── cpu/
│   │   ├── cpu-context.cc          ← [NEW]
│   │   ├── cpu-advector.cc         ← [NEW]
│   │   ├── cpu-projector.cc        ← [NEW]
│   │   ├── cpu-reconstructor.cc    ← [NEW]
│   │   ├── cpu-extrapolator.cc     ← [NEW]
│   │   ├── cpu-force.cc            ← [NEW]
│   │   ├── advect-solver.cc        ← 保留
│   │   ├── project-solver.cc       ← 保留
│   │   └── util.cc                 ← 保留
│   └── gpu/
│       ├── gpu-context.cc           ← [NEW]
│       ├── gpu-advector.cc          ← [KEEP] 适配
│       ├── gpu-projector.cc         ← [KEEP] 适配
│       ├── gpu-reconstructor.cc     ← [KEEP] 适配
│       ├── gpu-extrapolator.cc      ← [NEW]
│       └── gpu-force.cc             ← [NEW]
├── shaders/                         ← 保留全部 HLSL
└── CMakeLists.txt                   ← 修改
```

---

## 12. 与现有代码的关系

### 12.1 保留的内部实现

以下类**保持不变**，作为新 Solver 的**内部实现细节**：

| 现有类 | 新的使用方式 |
|--------|-------------|
| `HybridAdvectionSolver3D` | 被 `CPUAdvectorP2G` / `CPUAdvectorG2P` 内部使用 |
| `FlipAdvectionSolver3D` | 被 `HybridAdvectionSolver3D` 使用（不变） |
| `PicAdvector3D` | 被 `HybridAdvectionSolver3D` 使用（不变） |
| `FvmSolver3D` / `ProjectionSolver3D` | 被 `CPUProjector` 内部使用 |
| `CgSolver3D` + `Preconditioner3D` | 被 `FvmSolver3D` 使用（不变） |
| `NaiveReconstructor<Real, 3>` | 被 `CPUReconstructor` 内部使用 |
| `SDF<3>` | 作为 Context 注册表中的值 |
| GPU 的 `GPUPcgSolver` / `GPUJacobiSolver` | 被 `GPUProjector` 内部使用 |

### 12.2 删除/替代的接口

| 现有 | 处理方式 |
|------|---------|
| `FluidBackend`（5 虚方法） | **删除** |
| `CPUFluidBackend` | **删除**——逻辑拆分到各 CPU Solver |
| `GPUFluidBackend` | **删除**——逻辑拆分到各 GPU Solver |
| `FluidComputeBackend`（老的全 setter 接口） | **删除** |
| 老的 `FluidSimulator`（Animation 子类） | **重写**为编排器 |
| `cpu::FluidSimulator`（500+ 行大类） | **拆分**为 Context + 多个 Solver |

### 12.3 对外接口变化

| 场景 | 旧 API | 新 API |
|------|--------|--------|
| 创建模拟器 | `FluidSimulator sim(config); sim.setBackend(CPU);` | `auto sim = presets::createFreeSurface(cfg, CPU);` |
| 推进一步 | `sim.step(frame);` | `sim->step(frame);`（不变） |
| 读取粒子 | `backend->readbackParticles(frame);` | `sim->context().get<CPUParticleData>("particles")` |
| 运行时调参 | `backend->updateSolverConfig(cfg);` | `sim->findSolver("cpu_projector")->configure(props);` |

---

## 13. 迁移路径

### Phase 1: 基础设施 (1.5d)

| # | 任务 | 产出 |
|---|------|------|
| 1.1 | 创建 `fluid-context.h` / `.cc` | FluidContext 基类 + std::any 注册表 |
| 1.2 | 创建 `fluid-solver.h` | FluidModularSolver 基类 |
| 1.3 | 创建 `field-tags.h` | Well-known field constants |
| 1.4 | 重写 `fluid-simulator.h` / `.cc` | 新的 FluidSimulator（构造即就绪） |

### Phase 2: CPU Solver 拆分 (3d)

| # | 任务 | 产出 |
|---|------|------|
| 2.1 | 创建 `cpu-context.h` / `.cc` + `cpu-solver.h` | CPUFluidContext + CPUModularSolver |
| 2.2 | 创建 `cpu-advector.h` / `.cc` | CPUAdvectorP2G / CPUAdvectorG2P |
| 2.3 | 创建 `cpu-projector.h` / `.cc` | CPUProjector |
| 2.4 | 创建 `cpu-reconstructor.h` / `.cc` | CPUReconstructor |
| 2.5 | 创建 `cpu-extrapolator.h` / `.cc` + `cpu-force.h` / `.cc` | 剩余 |
| 2.6 | 创建 `presets.cc` — `createFreeSurface(CPU)` | 预设工厂 |
| 2.7 | 验证：新旧管线输出 bit-exact 一致 | 回归测试 |

### Phase 3: GPU Solver 适配 (2d)

| # | 任务 | 产出 |
|---|------|------|
| 3.1 | 创建 `gpu-context.h` / `.cc` + `gpu-solver.h` | GPUFluidContext + GPUModularSolver |
| 3.2 | 适配现有 GPUAdvector → GPUModularSolver | |
| 3.3 | 适配现有 GPUProjector → GPUModularSolver | |
| 3.4 | 新增 `gpu-extrapolator` / `gpu-force` | |
| 3.5 | `createFreeSurface(GPU)` | GPU 预设 |

### Phase 4: 清理 (1d)

| # | 任务 |
|---|------|
| 4.1 | 删除 `FluidBackend`、`CPUFluidBackend`、`GPUFluidBackend` |
| 4.2 | 删除老的 `FluidComputeBackend`、`cpu::FluidSimulator` |
| 4.3 | 更新 CMakeLists.txt |
| 4.4 | 更新 Python binding |

**总计**: ~7.5 天

---

## 附录

### 附录 A: 完整执行顺序

#### 自由液面 (FreeSurface FLIP)

```
solvers = [
    CPUAdvectorP2G,       // P2G scatter + write validity flags
    CPUExtrapolator,      // 速度外推 (读 validity flags from 注册表)
    CPUReconstructor,     // 粒子 → fluid SDF
    CPUForceApplicator,   // v += gravity * dt
    CPUProjector,         // 压力投影
    CPUAdvectorG2P,       // G2P gather + RK3 advect
]
```

CPUAdvectorP2G 和 CPUAdvectorG2P 共享同一个底层 `CPUAdvector` 实现对象，通过 `shared_ptr` 持有：

```cpp
auto advImpl = std::make_shared<cpu::CPUAdvector>(ctx, config.advector);
auto advP2G  = std::make_shared<cpu::CPUAdvectorP2G>(advImpl);
auto advG2P  = std::make_shared<cpu::CPUAdvectorG2P>(advImpl);
```

#### 烟雾 (Eulerian Smoke)

```
solvers = [
    DensityAdvector,      // 半拉格朗日平流 density/temperature
    BuoyancySolver,       // v += buoyancy force
    ForceApplicator,      // v += gravity * dt
    Projector,            // 压力投影
]
```

### 附录 B: 注册表定位

注册表 (`std::any` map) 在本设计中是一个**轻量级扩展点**，仅用于：
- 用户自定义物理量（如化学浓度场）
- 无法预知的实验性数据

所有常见的模拟数据（velocity、particles、SDF、validity flags、density、temperature）都是 Context 子类的**直接成员**，不走注册表。这意味着：
- 热路径上不会有 hash lookup 开销
- 大部分 Solver 代码只做 `ctx.u`、`ctx.positions` 等直接成员访问
- 注册表几乎只在自定义扩展场景中使用

### 附录 C: Python Binding 示意

```python
import simcraft.fluid as fluid

# 使用预设
config = fluid.FreeSurfaceConfig()
config.domain = fluid.FluidDomain(resolution=[64, 64, 64], size=[1, 1, 1])
config.advector.method = fluid.AdvectionMethod.FLIP
config.advector.flip_blend = 0.97

sim = fluid.presets.create_free_surface(config, backend="cpu")

# 推进模拟
for i in range(100):
    sim.step(dt=1.0/60.0)

# 读取结果
particles = sim.context().get("particles")  # CPUParticleData
positions = particles.positions  # numpy array

# 运行时调参
sim.find_solver("cpu_projector").configure({"max_iters": 500})
```

### 附录 D: 未来扩展示例

#### 添加自定义物理量

```cpp
class ChemicalDiffusionSolver : public cpu::CPUModularSolver {
public:
    ChemicalDiffusionSolver(cpu::CPUFluidContext& ctx, Config config)
        : CPUModularSolver("chemical_diffusion"), config_(config)
    {
        // 注册自定义物理量——Context 不需要提前知道
        ctx.ensure<spatify::Array3D<Real>>("chemical_concentration", nx, ny, nz);
    }

    void solve(cpu::CPUFluidContext& ctx, Real dt) override {
        auto& conc = ctx.get<spatify::Array3D<Real>>("chemical_concentration");
        // 扩散方程求解...
    }
};
```

### 附录 E: 与 fluid-gpu-design.md 的关系

本文档是对 `fluid-gpu-design.md` (v3.1) 的**架构升级**。v3.1 中的 CPU/GPU 内部组件设计（Shader 清单、数据布局、CG 实现等）**全部保留**，只是外层从 `FluidBackend` 变为 `FluidModularSolver` 子类。

| v3.1 概念 | 本文档对应 |
|-----------|-----------|
| `FluidBackend` 接口 | 删除，替代为 `FluidSimulator` + `FluidModularSolver` 组合 |
| `CPUFluidBackend` | 拆为 `CPUFluidContext` + 多个 `CPUModularSolver` |
| `GPUFluidBackend` | 拆为 `GPUFluidContext` + 多个 `GPUModularSolver` |
| `GPUGridState` | 移入 `GPUFluidContext` 作为固定成员 |
| `substep()` 编排 | 变为 `FluidSimulator::step()` 遍历 solver 列表 |
| Shader 设计 | 完全不变 |

---

## 附录 F: 关键风险与缓解

| 风险 | 严重度 | 缓解措施 |
|------|--------|---------|
| `std::any_cast` 类型不匹配 | 中 | 编译 Debug 模式有清晰错误信息；well-known tags 常量避免拼写错误 |
| CPU/GPU Solver 混搭 | 低 | 工厂函数保证配套；Debug 模式可加 `backendTag()` 断言 |
| Solver 执行顺序错误 | 中 | 预设工厂内硬编码正确顺序 |
| `std::any` 小对象优化不适用于大类型 | 低 | 大数据用 `unique_ptr` 包装后存入 any |
| GPU CommandList 生命周期 | 低 | `beginFrame()`/`endFrame()` 严格由 Simulator 管理 |
