//
// Created by creeper on 6/12/24.
//

#include <Core/json.h>
#include <Deform/invariants.h>
#include <fem/system.h>
#include <spdlog/spdlog.h>
#include <optional>

namespace ksk::fem {
void tetAssembleGlobal(VecXd &global, const Vector<Real, 12> &local,
                       const Vector<int, 4> &tet) {
  for (int i = 0; i < 4; i++)
    global.segment<3>(tet[i] * 3) += local.segment<3>(3 * i);
}

void System::spdProjectHessian(
    maths::BlockSparseMatrix<3> &blockH) const {
  autoDispatch([this, &blockH](const Primitive &pr, int id) {
    pr.assembleEnergyHessian(blockH, dofStarts[id] / 3);
  });
}

void System::buildBlockMassMatrix() {
  int numBlocks = static_cast<int>(dof() / 3);
  m_blockMass = maths::BlockSparseMatrix<3>(numBlocks, numBlocks);
  autoDispatch([this](const Primitive &pr, int id) {
    pr.assembleMassMatrix(m_blockMass, dofStarts[id] / 3);
  });
  m_blockMass.sortByRow();
}

Real System::deformationEnergy() const { return cachedEnergy; }

void System::updateCurrentConfig(const maths::BlockVector<3> &x_nxt) {
  x = x_nxt;
  updateDeformationGradient();
}



void System::updateDeformationGradient() {
  autoDispatch([this](Primitive &pr, int id) {
    pr.updateDeformationEnergyGradient(pr.view(x));
  });
  updateDeformationEnergy();
  updateDeformationEnergyGradient();
}

maths::BlockVector<3> symbolicDeformationEnergyGradient(System &system) {
  return system.deformationEnergyGradient();
}

maths::BlockVector<3> numericalDeformationEnergyGradient(System &system) {
  maths::BlockVector<3> grad(system.x.numBlocks());
  grad.setZero();
  Real dx = 1e-7;
  maths::BlockVector<3> saved = system.x;
  for (int i = 0; i < system.dof(); i++) {
    system.x.data()[i] += dx;
    system.updateCurrentConfig(system.x);
    Real E_plus = system.deformationEnergy();
    system.x.data()[i] -= 2.0 * dx;
    system.updateCurrentConfig(system.x);
    Real E_minus = system.deformationEnergy();
    system.x.data()[i] += dx; // restore
    grad.data()[i] = (E_plus - E_minus) / (2 * dx);
  }
  system.updateCurrentConfig(saved);
  return grad;
}

void System::updateDeformationEnergy() {
  cachedEnergy = 0.0;
  autoDispatch([this](Primitive &pr, int id) {
    cachedEnergy += pr.deformationEnergy();
  });
}
void System::updateDeformationEnergyGradient() {
  energyGradient.setZero();
  autoDispatch([this](Primitive &pr, int dofStart) {
    pr.assembleEnergyGradient(pr.view(energyGradient));
  });
}

System &System::init() {
  size_t dofDim{};
  autoDispatch(
      [this, &dofDim](const Primitive &pr, int id) { dofDim += pr.dofDim(); });
  assert(dofDim % 3 == 0 && "DOF dimension must be a multiple of 3 for BlockVector<3>");
  int numBlocks = static_cast<int>(dofDim / 3);
  x.resize(numBlocks);
  xdot.resize(numBlocks);
  X.resize(numBlocks);
  energyGradient.resize(numBlocks);
  dofStarts.resize(prs.size());
  int dofStart = 0;
  for (int i = 0; i < prs.size(); ++i) {
    dofStarts[i] = dofStart;
    prs[i].setDofStart(dofStart);
    dofStart += static_cast<int>(prs[i].dofDim());
  }
  autoDispatch([this](Primitive &pr, int id) {
    auto x = pr.view(this->x);
    auto X = pr.view(this->X);
    auto xdot = pr.view(this->xdot);
    pr.init(x, xdot, X);
  });
  buildBlockMassMatrix();
  initGeometryManager();

  if (nEdges > 0) {
    Real totalLength = 0.0;
    for (int i = 0; i < nEdges; ++i) {
      auto vertices = getGlobalEdge(i);
      auto diff = x[vertices.x] - x[vertices.y];
      totalLength += glm::length(diff);
    }
    m_meshLengthScale = totalLength / nEdges;
  }

  // 提取 lumped mass: m_i = trace(M_ii) / 3
  m_lumpedMass.resize(numBlocks);
  auto diagBlocks = m_blockMass.extractDiagonal();
  for (int i = 0; i < numBlocks; i++) {
    m_lumpedMass[i] = (diagBlocks[i][0][0] + diagBlocks[i][1][1] + diagBlocks[i][2][2]) / 3.0;
  }

  // 构建约束索引（此时 m_constraints 应已被 builder 填充）
  m_constraints.build(numBlocks);

  logSystemInfo();
  return *this;
}

void System::initGeometryManager() {
  m_geometryManager.collectGeometryReferences(prs, m_colliders);
  nTriangles = m_geometryManager.triangleCount();
  nEdges = m_geometryManager.edgeCount();
  nVertices = m_geometryManager.vertexCount();
}

void System::logSystemInfo() const {
  // log all the system info so that we can know all the information about the
  // system
  spdlog::info("System Info:");
  spdlog::info("Number of primitives: {}", prs.size());
  spdlog::info("Number of colliders: {}", m_colliders.size());
  spdlog::info("Number of triangles: {}", nTriangles);
  spdlog::info("Number of edges: {}", nEdges);
  spdlog::info("Number of vertices: {}", nVertices);
  spdlog::info("Mesh length scale: {}", m_meshLengthScale);
  for (int i = 0; i < prs.size(); ++i) {
    spdlog::info("Primitive {}: dofDim = {}, dofStart = {}", i, prs[i].dofDim(),
                 dofStarts[i]);
  }
}

int System::globalVertexToPrimitive(int globalIdx) const {
  return m_geometryManager.getVertexRef(globalIdx).primitiveId;
}

maths::BlockVector<3> System::computeAcceleration() const {
  maths::BlockVector<3> accel(x.numBlocks());
  accel.setZero();
  // 当前仅重力: a = g (不需要除以质量，因为 f=mg, a=f/m=g)
  for (int i = 0; i < x.numBlocks(); i++)
    accel[i] = m_gravity;
  return accel;
}

System SystemBuilder::build(const core::JsonNode &json) {
  if (!json.is<core::JsonDict>())
    throw std::runtime_error("SystemBuilder requires a JSON object");
  const auto &dict = json.as<core::JsonDict>();

  // 使用 REFLECT 宏反序列化基本配置
  auto cfg = core::deserialize<SystemConfig>(json);

  System system;
  system.prs = std::move(cfg.primitives);
  system.m_colliders = std::move(cfg.colliders);

  // 解析重力配置
  if (dict.contains("gravity")) {
    const auto &g = dict.at("gravity");
    if (g.is<core::JsonList>()) {
      const auto &gList = g.as<core::JsonList>();
      if (gList.size() >= 3) {
        system.setGravity(glm::dvec3(
            gList[0].as<Real>(),
            gList[1].as<Real>(),
            gList[2].as<Real>()));
      }
    }
  }

  // 暂存需要延迟处理的 pin 约束（需要等待 init() 后获取 positions）
  std::vector<int> pendingPinVertices;

  // 解析约束配置
  if (dict.contains("constraints")) {
    const auto &cList = dict.at("constraints").as<core::JsonList>();
    for (const auto &cNode : cList) {
      const auto &cDict = cNode.as<core::JsonDict>();
      auto type = cDict.at("type").as<std::string>();

      if (type == "pin") {
        // 延迟到 init() 后执行 (需要知道 positions)
        if (cDict.contains("vertices")) {
          const auto &vList = cDict.at("vertices").as<core::JsonList>();
          for (const auto &v : vList) {
            pendingPinVertices.push_back(v.as<int>());
          }
        }
      } else if (type == "pin_component") {
        int vertex = cDict.at("vertex").as<int>();
        Real value = 0.0;
        if (cDict.contains("value")) {
          value = cDict.at("value").as<Real>();
        }
        // 解析 components 数组 [x, y, z]
        glm::bvec3 components{false, false, false};
        if (cDict.contains("components")) {
          const auto &compList = cDict.at("components").as<core::JsonList>();
          if (compList.size() >= 3) {
            components.x = compList[0].as<bool>();
            components.y = compList[1].as<bool>();
            components.z = compList[2].as<bool>();
          }
        }
        // 添加分量约束
        for (int comp = 0; comp < 3; comp++) {
          if (components[comp]) {
            system.constraints().pinComponent(vertex, comp, value);
          }
        }
      } else if (type == "prescribed_motion") {
        // 时变位移约束
        if (cDict.contains("vertices") && cDict.contains("motion")) {
          const auto &vList = cDict.at("vertices").as<core::JsonList>();
          const auto &mDict = cDict.at("motion").as<core::JsonDict>();
          auto motionType = mDict.at("type").as<std::string>();

          if (motionType == "sinusoidal") {
            glm::dvec3 axis{0.0, 1.0, 0.0};
            Real amplitude = 0.1;
            Real frequency = 1.0;

            if (mDict.contains("axis")) {
              const auto &aList = mDict.at("axis").as<core::JsonList>();
              if (aList.size() >= 3) {
                axis = glm::dvec3(aList[0].as<Real>(), aList[1].as<Real>(), aList[2].as<Real>());
              }
            }
            if (mDict.contains("amplitude")) amplitude = mDict.at("amplitude").as<Real>();
            if (mDict.contains("frequency")) frequency = mDict.at("frequency").as<Real>();

            Real w = 2.0 * glm::pi<Real>() * frequency;
            glm::dvec3 dir = glm::normalize(axis);

            // 为每个顶点添加时变约束
            for (const auto &v : vList) {
              int vertexIdx = v.as<int>();
              // 注意：这里需要先获取初始位置，所以在 init() 后需要重新设置
              // 暂时使用 lambda 捕获 axis, amplitude, frequency, w, dir
              auto posFunc = [=](Real t) -> glm::dvec3 {
                // 初始位置会在 init() 后设置，这里先返回 0
                // 实际使用时应该在 init() 后重新设置约束
                return glm::dvec3(0.0);
              };
              auto velFunc = [=](Real t) -> glm::dvec3 {
                return dir * amplitude * w * std::cos(w * t);
              };
              system.constraints().prescribeMotion(vertexIdx, posFunc, velFunc);
            }
          }
        }
      }
    }
  }

  // 初始化系统（会调用 m_constraints.build()）
  system.init();

  // init() 之后 positions 已设置，可以处理 pin 约束
  if (!pendingPinVertices.empty()) {
    system.constraints().pinVertices(pendingPinVertices, system.x);
    system.constraints().build(system.x.numBlocks());  // 重建索引
  }

  // 重建约束索引（如果有约束被添加）
  if (!system.constraints().allConstraints().empty()) {
    system.constraints().build(system.x.numBlocks());
  }

  return system;
}  // closes SystemBuilder::build()

} // namespace ksk::fem
