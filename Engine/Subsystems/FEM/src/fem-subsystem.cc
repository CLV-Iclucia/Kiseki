#include <FEM/fem-subsystem.h>

#include <FEM/cpu/fem-cpu-backend.h>

#include <Core/profiler.h>
#include <Deform/deformation-gradient.h>
#include <Deform/strain-energy-density.h>
#include <Maths/tensor.h>

#include <glm/geometric.hpp>

#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ksk::engine::fem
{
    namespace
    {
        int vertexScalarCount(const TetMeshDesc& mesh)
        {
            return 3 * static_cast<int>(mesh.vertices.size());
        }

        int vertexScalarCount(const ClothMeshDesc& mesh)
        {
            return 3 * static_cast<int>(mesh.vertices.size());
        }

        glm::dvec3 initialPosition(const TetMeshDesc& mesh, int vertex)
        {
            if (!mesh.initialPositions.empty())
            {
                return mesh.initialPositions.at(static_cast<size_t>(vertex));
            }
            return mesh.vertices.at(static_cast<size_t>(vertex));
        }

        glm::dvec3 initialVelocity(const TetMeshDesc& mesh, int vertex)
        {
            if (!mesh.initialVelocities.empty())
            {
                return mesh.initialVelocities.at(static_cast<size_t>(vertex));
            }
            return glm::dvec3(0.0);
        }

        glm::dvec3 initialPosition(const ClothMeshDesc& mesh, int vertex)
        {
            if (!mesh.initialPositions.empty())
            {
                return mesh.initialPositions.at(static_cast<size_t>(vertex));
            }
            return mesh.vertices.at(static_cast<size_t>(vertex));
        }

        glm::dvec3 initialVelocity(const ClothMeshDesc& mesh, int vertex)
        {
            if (!mesh.initialVelocities.empty())
            {
                return mesh.initialVelocities.at(static_cast<size_t>(vertex));
            }
            return glm::dvec3(0.0);
        }

        std::vector<glm::dvec3> currentPositions(const TetMeshDesc& mesh)
        {
            if (!mesh.initialPositions.empty())
            {
                return mesh.initialPositions;
            }
            return mesh.vertices;
        }

        std::vector<glm::dvec3> currentPositions(const ClothMeshDesc& mesh)
        {
            if (!mesh.initialPositions.empty())
            {
                return mesh.initialPositions;
            }
            return mesh.vertices;
        }

        std::vector<std::array<int, 2>> clothEdges(const ClothMeshDesc& mesh)
        {
            if (!mesh.edges.empty())
            {
                return mesh.edges;
            }
            std::set<std::array<int, 2>> unique_edges;
            for (const auto& triangle : mesh.triangles)
            {
                for (const auto edge : {
                         std::array<int, 2>{triangle[0], triangle[1]},
                         std::array<int, 2>{triangle[1], triangle[2]},
                         std::array<int, 2>{triangle[2], triangle[0]}
                     })
                {
                    unique_edges.insert(edge[0] < edge[1]
                                            ? edge
                                            : std::array<int, 2>{edge[1], edge[0]});
                }
            }
            return {unique_edges.begin(), unique_edges.end()};
        }

        int propertyLane(const std::string& property)
        {
            if (property == "x")
            {
                return 0;
            }
            if (property == "y")
            {
                return 1;
            }
            if (property == "z")
            {
                return 2;
            }
            throw std::runtime_error("FEM constraint property is not supported: " +
                property);
        }

        deform::StableNeoHookean<double> createEnergy(const TetMaterial& material)
        {
            return deform::StableNeoHookean<double>(
                deform::ElasticityParameters<double>{
                    .E = material.youngsModulus,
                    .nu = material.poissonRatio,
                });
        }

        Eigen::Matrix3d tetEdges(const TetMeshDesc& mesh,
                                 const Eigen::VectorXd& localQ,
                                 int meshBase,
                                 const std::array<int, 4>& tet)
        {
            Eigen::Matrix3d edges;
            const int root = meshBase + 3 * tet[0];
            for (int column = 0; column < 3; ++column)
            {
                const int vertex = meshBase + 3 * tet[column + 1];
                edges.col(column) =
                    localQ.segment<3>(vertex) - localQ.segment<3>(root);
            }
            return edges;
        }

        Eigen::Matrix3d restTetEdges(const TetMeshDesc& mesh,
                                     const std::array<int, 4>& tet)
        {
            Eigen::Matrix3d edges;
            const glm::dvec3 root = mesh.vertices.at(static_cast<size_t>(tet[0]));
            for (int column = 0; column < 3; ++column)
            {
                const glm::dvec3 vertex =
                    mesh.vertices.at(static_cast<size_t>(tet[column + 1]));
                const glm::dvec3 edge = vertex - root;
                edges.col(column) = Eigen::Vector3d(edge.x, edge.y, edge.z);
            }
            return edges;
        }

        using Matrix3x2 = Eigen::Matrix<double, 3, 2>;
        using Matrix6x6 = Eigen::Matrix<double, 6, 6>;
        using Matrix9x6 = Eigen::Matrix<double, 9, 6>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;
        using Vector9 = Eigen::Matrix<double, 9, 1>;

        Eigen::Vector3d toEigen(const glm::dvec3& value)
        {
            return Eigen::Vector3d(value.x, value.y, value.z);
        }

        Vector6 flatten(const Matrix3x2& value)
        {
            Vector6 result;
            result.segment<3>(0) = value.col(0);
            result.segment<3>(3) = value.col(1);
            return result;
        }

        struct ClothTriangleKinematics
        {
            Matrix3x2 F = Matrix3x2::Zero();
            Matrix9x6 pFpx = Matrix9x6::Zero();
            double area = 0.0;
        };

        std::optional<ClothTriangleKinematics> clothTriangleKinematics(
            const ClothMeshDesc& mesh,
            const Eigen::VectorXd& localQ,
            int meshBase,
            const std::array<int, 3>& triangle)
        {
            const Eigen::Vector3d X0 =
                toEigen(mesh.vertices.at(static_cast<size_t>(triangle[0])));
            const Eigen::Vector3d X1 =
                toEigen(mesh.vertices.at(static_cast<size_t>(triangle[1])));
            const Eigen::Vector3d X2 =
                toEigen(mesh.vertices.at(static_cast<size_t>(triangle[2])));
            Eigen::Vector3d axis_u = X1 - X0;
            const double length_u = axis_u.norm();
            if (length_u <= 1.0e-12)
            {
                return std::nullopt;
            }
            axis_u /= length_u;
            Eigen::Vector3d normal = axis_u.cross(X2 - X0);
            const double normal_length = normal.norm();
            if (normal_length <= 1.0e-12)
            {
                return std::nullopt;
            }
            normal /= normal_length;
            const Eigen::Vector3d axis_v = normal.cross(axis_u);

            const Eigen::Vector2d U0(0.0, 0.0);
            const Eigen::Vector2d U1((X1 - X0).dot(axis_u),
                                     (X1 - X0).dot(axis_v));
            const Eigen::Vector2d U2((X2 - X0).dot(axis_u),
                                     (X2 - X0).dot(axis_v));
            Eigen::Matrix2d Dm;
            Dm.col(0) = U1 - U0;
            Dm.col(1) = U2 - U0;
            const double det = Dm.determinant();
            if (std::abs(det) <= 1.0e-12)
            {
                return std::nullopt;
            }

            const Eigen::Vector3d x0 =
                localQ.segment<3>(meshBase + 3 * triangle[0]);
            const Eigen::Vector3d x1 =
                localQ.segment<3>(meshBase + 3 * triangle[1]);
            const Eigen::Vector3d x2 =
                localQ.segment<3>(meshBase + 3 * triangle[2]);
            Matrix3x2 Ds;
            Ds.col(0) = x1 - x0;
            Ds.col(1) = x2 - x0;

            const Eigen::Matrix2d inv_Dm = Dm.inverse();
            ClothTriangleKinematics result;
            result.F = Ds * inv_Dm;
            result.area = 0.5 * std::abs(det);

            const double c0 = -inv_Dm(0, 0) - inv_Dm(1, 0);
            const double c1 = inv_Dm(0, 0);
            const double c2 = inv_Dm(1, 0);
            const double d0 = -inv_Dm(0, 1) - inv_Dm(1, 1);
            const double d1 = inv_Dm(0, 1);
            const double d2 = inv_Dm(1, 1);
            const std::array<double, 3> column_u{c0, c1, c2};
            const std::array<double, 3> column_v{d0, d1, d2};
            for (int vertex = 0; vertex < 3; ++vertex)
            {
                for (int lane = 0; lane < 3; ++lane)
                {
                    result.pFpx(3 * vertex + lane, lane) = column_u[vertex];
                    result.pFpx(3 * vertex + lane, 3 + lane) = column_v[vertex];
                }
            }
            return result;
        }

        Matrix6x6 stretchHessian(const Matrix3x2& F, double mu)
        {
            Matrix6x6 H = Matrix6x6::Zero();
            const double I5u = F.col(0).squaredNorm();
            const double I5v = F.col(1).squaredNorm();
            if (I5u <= 1.0e-24 || I5v <= 1.0e-24)
            {
                return H;
            }
            const double invSqrtI5u = 1.0 / std::sqrt(I5u);
            const double invSqrtI5v = 1.0 / std::sqrt(I5v);
            H(0, 0) = H(1, 1) = H(2, 2) =
                std::max(1.0 - invSqrtI5u, 0.0);
            H(3, 3) = H(4, 4) = H(5, 5) =
                std::max(1.0 - invSqrtI5v, 0.0);

            const Eigen::Vector3d fu = F.col(0).normalized();
            const double u_coeff =
                (1.0 - invSqrtI5u >= 0.0) ? invSqrtI5u : 1.0;
            H.block<3, 3>(0, 0) += u_coeff * (fu * fu.transpose());

            const Eigen::Vector3d fv = F.col(1).normalized();
            const double v_coeff =
                (1.0 - invSqrtI5v >= 0.0) ? invSqrtI5v : 1.0;
            H.block<3, 3>(3, 3) += v_coeff * (fv * fv.transpose());
            return mu * H;
        }

        Matrix6x6 shearHessian(const Matrix3x2& F, double mu)
        {
            const double I6 = F.col(0).dot(F.col(1));
            const double signI6 = (I6 >= 0.0) ? 1.0 : -1.0;
            Matrix6x6 H = Matrix6x6::Zero();
            H(3, 0) = H(4, 1) = H(5, 2) = 1.0;
            H(0, 3) = H(1, 4) = H(2, 5) = 1.0;

            Matrix3x2 shear_direction = Matrix3x2::Zero();
            shear_direction.col(0) = F.col(1);
            shear_direction.col(1) = F.col(0);
            const Vector6 g = flatten(shear_direction);
            const double I2 = F.squaredNorm();
            const double lambda0 =
                0.5 * (I2 + std::sqrt(I2 * I2 + 12.0 * I6 * I6));
            const Vector6 candidate = I6 * H * g + lambda0 * g;
            if (candidate.squaredNorm() <= 1.0e-24)
            {
                return Matrix6x6::Zero();
            }
            const Vector6 q0 = candidate.normalized();
            const Matrix6x6 T =
                0.5 * (Matrix6x6::Identity() + signI6 * H);
            const Vector6 Tq = T * q0;
            const double normTq = Tq.squaredNorm();
            if (normTq <= 1.0e-24)
            {
                return mu * lambda0 * (q0 * q0.transpose());
            }
            return mu * (std::abs(I6) *
                             (T - (Tq * Tq.transpose()) / normTq) +
                         lambda0 * (q0 * q0.transpose()));
        }

        Vector6 clothEnergyGradientWrtF(const Matrix3x2& F, double mu)
        {
            Vector6 gradient = Vector6::Zero();
            const double lu = F.col(0).norm();
            const double lv = F.col(1).norm();
            if (lu > 1.0e-12)
            {
                gradient.segment<3>(0) +=
                    mu * (lu - 1.0) * F.col(0) / lu;
            }
            if (lv > 1.0e-12)
            {
                gradient.segment<3>(3) +=
                    mu * (lv - 1.0) * F.col(1) / lv;
            }
            const double shear = F.col(0).dot(F.col(1));
            gradient.segment<3>(0) += mu * shear * F.col(1);
            gradient.segment<3>(3) += mu * shear * F.col(0);
            return gradient;
        }

        double clothTriangleEnergy(const Matrix3x2& F, double mu)
        {
            const double stretch_u = F.col(0).norm() - 1.0;
            const double stretch_v = F.col(1).norm() - 1.0;
            const double shear = F.col(0).dot(F.col(1));
            return 0.5 * mu *
                (stretch_u * stretch_u + stretch_v * stretch_v +
                 shear * shear);
        }

        double clothEnergy(const ClothMeshDesc& mesh,
                           const Eigen::VectorXd& localQ,
                           int meshBase)
        {
            double energy = 0.0;
            for (const auto& triangle : mesh.triangles)
            {
                const auto kinematics =
                    clothTriangleKinematics(mesh, localQ, meshBase, triangle);
                if (!kinematics)
                {
                    continue;
                }
                energy += kinematics->area *
                    clothTriangleEnergy(
                        kinematics->F, mesh.material.stretchStiffness);
            }
            return energy;
        }

        void addClothGradient(const ClothMeshDesc& mesh,
                              const Eigen::VectorXd& localQ,
                              int meshBase,
                              Eigen::VectorXd& gradient)
        {
            for (const auto& triangle : mesh.triangles)
            {
                const auto kinematics = clothTriangleKinematics(mesh, localQ, meshBase, triangle);
                if (!kinematics)
                {
                    continue;
                }
                const Vector9 local_gradient =
                    kinematics->area * kinematics->pFpx *
                    clothEnergyGradientWrtF(
                        kinematics->F, mesh.material.stretchStiffness);
                for (int vertex = 0; vertex < 3; ++vertex)
                {
                    gradient.segment<3>(meshBase + 3 * triangle[vertex]) +=
                        local_gradient.segment<3>(3 * vertex);
                }
            }
        }

        void addClothHessian(const ClothMeshDesc& mesh,
                             const Eigen::VectorXd& localQ,
                             int meshBase,
                             std::vector<Eigen::Triplet<double>>& triplets)
        {
            SIM_PROFILE_SCOPE_COLOR("FEMSubsystem/AddClothHessian",
                                    ksk::core::profiler_colors::kCyan);
            for (const auto& triangle : mesh.triangles)
            {
                SIM_PROFILE_SCOPE("FEMSubsystem/AddClothHessian/Triangle");
                const auto kinematics =
                    clothTriangleKinematics(mesh, localQ, meshBase, triangle);
                if (!kinematics)
                {
                    continue;
                }

                Matrix6x6 hessian_F;
                {
                    SIM_PROFILE_SCOPE("FEMSubsystem/AddClothHessian/FHessian");
                    hessian_F =
                        stretchHessian(kinematics->F,
                                       mesh.material.stretchStiffness) +
                        shearHessian(kinematics->F,
                                     mesh.material.stretchStiffness);
                }

                Eigen::Matrix<double, 9, 9> local_hessian;
                {
                    SIM_PROFILE_SCOPE("FEMSubsystem/AddClothHessian/Local9x9");
                    local_hessian =
                        kinematics->area * kinematics->pFpx * hessian_F *
                        kinematics->pFpx.transpose();
                }

                {
                    SIM_PROFILE_SCOPE("FEMSubsystem/AddClothHessian/EmitTriplets");
                    for (int row_vertex = 0; row_vertex < 3; ++row_vertex)
                    {
                        for (int col_vertex = 0; col_vertex < 3; ++col_vertex)
                        {
                            for (int row_lane = 0; row_lane < 3; ++row_lane)
                            {
                                for (int col_lane = 0; col_lane < 3; ++col_lane)
                                {
                                    const double value =
                                        local_hessian(3 * row_vertex + row_lane,
                                                      3 * col_vertex + col_lane);
                                    if (value != 0.0)
                                    {
                                        triplets.emplace_back(
                                            meshBase + 3 * triangle[row_vertex] +
                                                row_lane,
                                            meshBase + 3 * triangle[col_vertex] +
                                                col_lane,
                                            value);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        std::vector<FEMPrimitive> makeTetPrimitives(std::vector<TetMeshDesc> meshes)
        {
            std::vector<FEMPrimitive> primitives;
            primitives.reserve(meshes.size());
            for (TetMeshDesc& mesh : meshes)
            {
                primitives.push_back(FEMTetMeshPrimitive{
                    .mesh = std::move(mesh),
                    .offset = {},
                    .runtime = {},
                });
            }
            return primitives;
        }
    } // namespace

    FEMSubsystem::FEMSubsystem(runtime::SubsystemId id,
                               std::vector<TetMeshDesc> meshes,
                               std::vector<FEMConstraintBinding> constraints,
                               int scalarOffset,
                               glm::dvec3 gravity)
        : FEMSubsystem(id,
                       makeTetPrimitives(std::move(meshes)),
                       std::move(constraints),
                       scalarOffset,
                       gravity)
    {
    }

    FEMSubsystem::FEMSubsystem(runtime::SubsystemId id,
                               std::vector<FEMPrimitive> primitives,
                               std::vector<FEMConstraintBinding> constraints,
                               int scalarOffset,
                               glm::dvec3 gravity)
        : id_(id)
          , gravity_(gravity)
          , primitives_(std::move(primitives))
          , constraints_(std::move(constraints))
    {
        int scalar_count = 0;
        int sample_count = 0;
        for (FEMPrimitive& primitive : primitives_)
        {
            std::visit([&](auto& value)
            {
                value.offset = FEMMeshOffset{
                    .q = scalar_count,
                    .samples = sample_count,
                };
                scalar_count += vertexScalarCount(value.mesh);
                sample_count += static_cast<int>(value.mesh.vertices.size());
            }, primitive);
        }

        range_ = runtime::DofRange{
            .subsystem = id_,
            .scalarOffset = scalarOffset,
            .scalarCount = scalar_count,
            .blockSize = 3,
        };
        rebuildSamples();
        rebuildCompatibilityViews();
        updateConstraintTargets(0.0);
        cpu_backend_ = std::make_unique<FEMCPUBackend>(*this);
    }

    FEMSubsystem::~FEMSubsystem() = default;

    runtime::SubsystemId FEMSubsystem::id() const noexcept
    {
        return id_;
    }

    runtime::DofRange FEMSubsystem::dofRange() const noexcept
    {
        return range_;
    }

    void FEMSubsystem::declareGeometry(runtime::GlobalGeometryManager& geometry)
    {
        SIM_PROFILE_SCOPE("FEMSubsystem/DeclareGeometry");
        geometry_points_.clear();
        geometry_points_.reserve(samples_.size());
        runtime_meshes_.clear();
        runtime_meshes_.reserve(primitives_.size());

        for (int mesh_index = 0; mesh_index < primitives_.size(); mesh_index++)
        {
            std::visit([&](auto& primitive)
            {
                const auto& mesh = primitive.mesh;
                const int sample_base = primitive.offset.samples;
                const std::vector<glm::dvec3> positions = currentPositions(mesh);
                const auto edges = [&]()
                {
                    if constexpr (std::is_same_v<std::decay_t<decltype(primitive)>,
                                                 FEMClothPrimitive>)
                    {
                        return clothEdges(mesh);
                    }
                    else
                    {
                        return mesh.surfaceEdges;
                    }
                }();
                const auto triangles = [&]()
                {
                    if constexpr (std::is_same_v<std::decay_t<decltype(primitive)>,
                                                 FEMClothPrimitive>)
                    {
                        return mesh.triangles;
                    }
                    else
                    {
                        return mesh.surfaceTriangles;
                    }
                }();
                const auto tets = [&]()
                {
                    if constexpr (std::is_same_v<std::decay_t<decltype(primitive)>,
                                                 FEMClothPrimitive>)
                    {
                        return std::vector<std::array<int, 4>>{};
                    }
                    else
                    {
                        return mesh.tets;
                    }
                }();
                const double thickness = [&]()
                {
                    if constexpr (std::is_same_v<std::decay_t<decltype(primitive)>,
                                                 FEMClothPrimitive>)
                    {
                        return mesh.material.thickness;
                    }
                    else
                    {
                        return 0.0;
                    }
                }();
                runtime::GeometryTransferMap transfer = runtime::transferGeometry(
                    geometry,
                    runtime::GeometryTransferInput{
                        .subsystem = id_,
                        .localInstanceId = mesh_index,
                        .localSampleOffset = sample_base,
                        .vertices = mesh.vertices,
                        .positions = positions,
                        .edges = edges,
                        .triangles = triangles,
                        .tets = tets,
                        .radius = thickness,
                    });
                geometry_points_.insert(geometry_points_.end(),
                                        transfer.localToGlobalPoint.begin(),
                                        transfer.localToGlobalPoint.end());
                primitive.runtime = FEMMeshRuntimeRef{
                    .points = transfer.points,
                    .surfaceEdges = transfer.edges,
                    .surfaceTriangles = transfer.triangles,
                    .tets = transfer.tets,
                    .transfer = std::move(transfer),
                    .material = {},
                };
                runtime_meshes_.push_back(primitive.runtime);
            }, primitives_[mesh_index]);
        }
    }

    void FEMSubsystem::writeState(runtime::DofView q,
                                  runtime::DofView qdot) const
    {
        cpu_backend_->writeState(q, qdot);
    }

    void FEMSubsystem::readState(runtime::ConstDofView q,
                                 runtime::ConstDofView qdot)
    {
        cpu_backend_->readState(q, qdot);
    }

    void FEMSubsystem::beginStep(runtime::ConstDofView q,
                                 runtime::ConstDofView qdot,
                                 double dt)
    {
        cpu_backend_->beginStep(q, qdot, dt);
    }

    void FEMSubsystem::acceptStep(runtime::ConstDofView q,
                                  runtime::DofView qdot,
                                  double dt)
    {
        cpu_backend_->acceptStep(q, qdot, dt);
    }

    double FEMSubsystem::evaluateObjective(runtime::ConstDofView q,
                                           runtime::ConstDofView qdot,
                                           double dt)
    {
        return cpu_backend_->evaluateObjective(q, qdot, dt);
    }

    void FEMSubsystem::updateInternalConstraints(double time, double dt)
    {
        cpu_backend_->updateInternalConstraints(time, dt);
    }

    void FEMSubsystem::prepareLocalOperator(double dt)
    {
        cpu_backend_->prepareLocalOperator(dt);
    }

    void FEMSubsystem::assembleLocalGradient(runtime::DofView g) const
    {
        cpu_backend_->assembleLocalGradient(g);
    }

    void FEMSubsystem::applyLocalMatrix(runtime::ConstDofView x,
                                        runtime::DofView y) const
    {
        cpu_backend_->applyLocalMatrix(x, y);
    }

    void FEMSubsystem::solveLocalSystem(runtime::ConstDofView b,
                                        runtime::DofView x) const
    {
        cpu_backend_->solveLocalSystem(b, x);
    }

    void FEMSubsystem::updateGeometry(runtime::GlobalGeometryManager& geometry) const
    {
        SIM_PROFILE_SCOPE("FEMSubsystem/UpdateGeometry");
        for (int sample_index = 0; sample_index < static_cast<int>(samples_.size());
             ++sample_index)
        {
            const runtime::PointIdx point = geometry_points_.at(sample_index);
            if (!geometry.contains(point))
            {
                throw std::runtime_error("FEM geometry point id is stale");
            }
            const FEMVertexSample& sample = samples_[sample_index];
            geometry.setPointPosition(point, vertexPosition(sample.mesh, sample.vertex));
            geometry.setPointRestPosition(
                point, restVertexPosition(sample.mesh, sample.vertex));
        }
    }

    void FEMSubsystem::mapLocalDirectionToGeometry(runtime::ConstDofView localDq,
                                                   runtime::GeometryView globalDx) const
    {
        cpu_backend_->mapLocalDirectionToGeometry(localDq, globalDx);
    }

    void FEMSubsystem::applyInternalContacts(runtime::ContactStencils contacts)
    {
        // The CPU backend consumes these routed internal contacts when assembling
        // contact energy, gradient, and Hessian products.
        internal_contacts_ = std::move(contacts);
    }

    void FEMSubsystem::scatterContactGradient(
        std::span<const runtime::PointIdx> points,
        runtime::ConstGeometryView pointGradient,
        runtime::DofView g) const
    {
        cpu_backend_->scatterContactGradient(points, pointGradient, g);
    }

    void FEMSubsystem::applyContactGeometryHessianProduct(
        std::span<const runtime::PointIdx> gradientPoints,
        runtime::ConstGeometryView pointGradient,
        std::span<const runtime::PointIdx> productPoints,
        runtime::ConstGeometryView pointHessianProduct,
        runtime::ConstDofView localDq,
        runtime::DofView localY) const
    {
        cpu_backend_->applyContactGeometryHessianProduct(
            gradientPoints,
            pointGradient,
            productPoints,
            pointHessianProduct,
            localDq,
            localY);
    }

    void FEMSubsystem::applyInternalContactHessian(runtime::ConstDofView localDq,
                                                   runtime::DofView localY) const
    {
        cpu_backend_->applyInternalContactHessian(localDq, localY);
    }

    void FEMSubsystem::visit(runtime::CpuSubsystemBackend& backend)
    {
    }

    void FEMSubsystem::visit(runtime::GpuSubsystemBackend& backend)
    {
    }

    int FEMSubsystem::localScalarCount() const noexcept
    {
        return range_.scalarCount;
    }

    void FEMSubsystem::rebuildSamples()
    {
        samples_.clear();
        samples_.reserve(static_cast<size_t>(range_.scalarCount / 3));
        for (int mesh_index = 0; mesh_index < static_cast<int>(primitives_.size());
             ++mesh_index)
        {
            std::visit([&](const auto& primitive)
            {
                const int base = primitive.offset.q;
                for (int vertex = 0;
                     vertex < static_cast<int>(primitive.mesh.vertices.size());
                     ++vertex)
                {
                    samples_.push_back(FEMVertexSample{
                        .mesh = mesh_index,
                        .vertex = vertex,
                        .qOffset = base + 3 * vertex,
                    });
                }
            }, primitives_[mesh_index]);
        }
    }

    void FEMSubsystem::rebuildCompatibilityViews()
    {
        compatibility_meshes_.clear();
        compatibility_meshes_.reserve(primitives_.size());
        for (const FEMPrimitive& primitive : primitives_)
        {
            if (const auto* tet = std::get_if<FEMTetMeshPrimitive>(&primitive))
            {
                compatibility_meshes_.push_back(tet->mesh);
            }
        }
    }

    void FEMSubsystem::updateConstraintTargets(double time)
    {
        active_constraints_.clear();
        active_constraints_.reserve(constraints_.size());
        for (const FEMConstraintBinding& binding : constraints_)
        {
            const int offset = constraintLocalOffset(binding);
            if (!binding.constraint.target)
            {
                throw std::runtime_error("FEM constraint requires a scalar target");
            }
            active_constraints_.push_back(ActiveConstraint{
                .qOffset = offset,
                .stiffness = binding.constraint.stiffness,
                .target = binding.constraint.target(time),
            });
        }
    }

    Eigen::VectorXd FEMSubsystem::gatherCurrentState() const
    {
        SIM_PROFILE_SCOPE("FEMSubsystem/GatherCurrentState");
        Eigen::VectorXd values(range_.scalarCount);
        for (const FEMVertexSample& sample : samples_)
        {
            const glm::dvec3 position = vertexPosition(sample.mesh, sample.vertex);
            values[sample.qOffset + 0] = position.x;
            values[sample.qOffset + 1] = position.y;
            values[sample.qOffset + 2] = position.z;
        }
        return values;
    }

    double FEMSubsystem::elasticEnergy(const Eigen::VectorXd& localQ) const
    {
        SIM_PROFILE_SCOPE_COLOR("FEMSubsystem/ElasticEnergy",
                                ksk::core::profiler_colors::kPurple);
        double energy = 0.0;
        for (int mesh_index = 0; mesh_index < static_cast<int>(primitives_.size());
             ++mesh_index)
        {
            std::visit([&](const auto& primitive)
            {
                using Primitive = std::decay_t<decltype(primitive)>;
                const auto& mesh = primitive.mesh;
                const int mesh_base = primitive.offset.q;
                if constexpr (std::is_same_v<Primitive, FEMTetMeshPrimitive>)
                {
                    const auto model = createEnergy(mesh.material);
                    for (const auto& tet : mesh.tets)
                    {
                        const Eigen::Matrix3d local_X = restTetEdges(mesh, tet);
                        const double volume = local_X.determinant() / 6.0;
                        if (volume <= 0.0)
                        {
                            throw std::runtime_error("FEM tet has non-positive rest volume");
                        }
                        deform::DeformationGradient<double, 3> dg(local_X);
                        dg.updateCurrentConfig(tetEdges(mesh, localQ, mesh_base, tet));
                        energy += model.computeEnergyDensity(dg) * volume;
                    }
                }
                else
                {
                    energy += clothEnergy(mesh, localQ, mesh_base);
                }
            }, primitives_[mesh_index]);
        }
        return energy;
    }

    void FEMSubsystem::assembleElasticGradient(const Eigen::VectorXd& localQ,
                                               Eigen::VectorXd& gradient) const
    {
        SIM_PROFILE_SCOPE_COLOR("FEMSubsystem/ElasticGradient",
                                ksk::core::profiler_colors::kRed);
        for (int mesh_index = 0; mesh_index < static_cast<int>(primitives_.size());
             ++mesh_index)
        {
            std::visit([&](const auto& primitive)
            {
                using Primitive = std::decay_t<decltype(primitive)>;
                const auto& mesh = primitive.mesh;
                const int mesh_base = primitive.offset.q;
                if constexpr (std::is_same_v<Primitive, FEMTetMeshPrimitive>)
                {
                    const auto model = createEnergy(mesh.material);
                    for (const auto& tet : mesh.tets)
                    {
                        const Eigen::Matrix3d local_X = restTetEdges(mesh, tet);
                        const double volume = local_X.determinant() / 6.0;
                        if (volume <= 0.0)
                        {
                            throw std::runtime_error("FEM tet has non-positive rest volume");
                        }
                        deform::DeformationGradient<double, 3> dg(local_X);
                        dg.updateCurrentConfig(tetEdges(mesh, localQ, mesh_base, tet));
                        const Eigen::Matrix<double, 12, 1> local_gradient =
                            dg.gradient().transpose() *
                            maths::vectorize(model.computeEnergyGradient(dg)) * volume;
                        for (int vertex = 0; vertex < 4; ++vertex)
                        {
                            gradient.segment<3>(mesh_base + 3 * tet[vertex]) +=
                                local_gradient.segment<3>(3 * vertex);
                        }
                    }
                }
                else
                {
                    addClothGradient(mesh, localQ, mesh_base, gradient);
                }
            }, primitives_[mesh_index]);
        }
    }

    void FEMSubsystem::assembleElasticHessian(
        const Eigen::VectorXd& localQ,
        std::vector<Eigen::Triplet<double>>& triplets) const
    {
        SIM_PROFILE_SCOPE_COLOR("FEMSubsystem/ElasticHessian",
                                ksk::core::profiler_colors::kCyan);
        for (const auto & pr : primitives_)
        {
            std::visit([&](const auto& primitive)
            {
                using Primitive = std::decay_t<decltype(primitive)>;
                const auto& mesh = primitive.mesh;
                const int mesh_base = primitive.offset.q;
                if constexpr (std::is_same_v<Primitive, FEMTetMeshPrimitive>)
                {
                    const auto model = createEnergy(mesh.material);
                    for (const auto& tet : mesh.tets)
                    {
                        const Eigen::Matrix3d local_X = restTetEdges(mesh, tet);
                        const double volume = local_X.determinant() / 6.0;
                        if (volume <= 0.0)
                        {
                            throw std::runtime_error("FEM tet has non-positive rest volume");
                        }
                        deform::DeformationGradient<double, 3> dg(local_X);
                        dg.updateCurrentConfig(tetEdges(mesh, localQ, mesh_base, tet));
                        const Eigen::Matrix<double, 12, 12> local_hessian =
                            dg.gradient().transpose() * model.filteredEnergyHessian(dg) *
                            dg.gradient() * volume;
                        for (int row_vertex = 0; row_vertex < 4; ++row_vertex)
                        {
                            for (int col_vertex = 0; col_vertex < 4; ++col_vertex)
                            {
                                for (int row_lane = 0; row_lane < 3; ++row_lane)
                                {
                                    for (int col_lane = 0; col_lane < 3; ++col_lane)
                                    {
                                        const int row = mesh_base + 3 * tet[row_vertex] + row_lane;
                                        const int col = mesh_base + 3 * tet[col_vertex] + col_lane;
                                        const double value =
                                            local_hessian(3 * row_vertex + row_lane,
                                                          3 * col_vertex + col_lane);
                                        if (value != 0.0)
                                        {
                                            triplets.emplace_back(row, col, value);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    addClothHessian(mesh, localQ, mesh_base, triplets);
                }
            }, pr);
        }
    }

    glm::dvec3 FEMSubsystem::vertexPosition(int mesh, int vertex) const
    {
        return std::visit([&](const auto& primitive)
        {
            return initialPosition(primitive.mesh, vertex);
        }, primitives_.at(static_cast<size_t>(mesh)));
    }

    glm::dvec3 FEMSubsystem::restVertexPosition(int mesh, int vertex) const
    {
        return std::visit([&](const auto& primitive)
        {
            return primitive.mesh.vertices.at(static_cast<size_t>(vertex));
        }, primitives_.at(static_cast<size_t>(mesh)));
    }

    void FEMSubsystem::setVertexPosition(int mesh,
                                         int vertex,
                                         const glm::dvec3& position)
    {
        std::visit([&](auto& primitive)
        {
            auto& desc = primitive.mesh;
            if (desc.initialPositions.empty())
            {
                desc.initialPositions = desc.vertices;
            }
            desc.initialPositions.at(static_cast<size_t>(vertex)) = position;
        }, primitives_.at(static_cast<size_t>(mesh)));
    }

    glm::dvec3 FEMSubsystem::vertexVelocity(int mesh, int vertex) const
    {
        return std::visit([&](const auto& primitive)
        {
            return initialVelocity(primitive.mesh, vertex);
        }, primitives_.at(static_cast<size_t>(mesh)));
    }

    void FEMSubsystem::setVertexVelocity(int mesh,
                                         int vertex,
                                         const glm::dvec3& velocity)
    {
        std::visit([&](auto& primitive)
        {
            auto& desc = primitive.mesh;
            if (desc.initialVelocities.empty())
            {
                desc.initialVelocities.resize(desc.vertices.size(), glm::dvec3(0.0));
            }
            desc.initialVelocities.at(static_cast<size_t>(vertex)) = velocity;
        }, primitives_.at(static_cast<size_t>(mesh)));
    }

    int FEMSubsystem::constraintLocalOffset(
        const FEMConstraintBinding& binding) const
    {
        const FEMPrimitive& primitive = primitives_.at(static_cast<size_t>(binding.mesh));
        const int lane = propertyLane(binding.constraint.property);
        return std::visit([&](const auto& value)
        {
            (void)value;
            return value.offset.q + 3 * binding.constraint.sample + lane;
        }, primitive);
    }
} // namespace ksk::engine::fem
