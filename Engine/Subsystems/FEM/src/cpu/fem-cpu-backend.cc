#include <FEM/cpu/fem-cpu-backend.h>

#include <FEM/fem-subsystem.h>

#include <Contact/contact-barrier.h>

#include <Eigen/SparseCholesky>

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ksk::engine::fem
{
    namespace
    {
        void requireCPU(runtime::ConstDofView view, const char* operation)
        {
            if (!view.isCPU())
            {
                throw std::runtime_error(std::string(operation) +
                    " requires a CPU DofView");
            }
        }

        void requireCPU(runtime::DofView view, const char* operation)
        {
            requireCPU(view.asConst(), operation);
        }

        void requireCPU(runtime::ConstGeometryView view, const char* operation)
        {
            if (!view.isCPU())
            {
                throw std::runtime_error(std::string(operation) +
                    " requires a CPU GeometryView");
            }
        }

        void requireCPU(runtime::GeometryView view, const char* operation)
        {
            requireCPU(view.asConst(), operation);
        }

        void requireDofCount(runtime::ConstDofView view,
                             int expected,
                             const char* operation)
        {
            if (view.scalarCount() < expected)
            {
                throw std::invalid_argument(std::string(operation) +
                    " received a DOF view that is too small");
            }
        }

        Eigen::VectorXd gatherLocalVector(runtime::ConstDofView view,
                                          int scalarCount)
        {
            requireDofCount(view, scalarCount, "gatherLocalVector");
            Eigen::VectorXd local(scalarCount);
            for (int i = 0; i < scalarCount; ++i)
            {
                local[i] = view[i];
            }
            return local;
        }

        int contactPointCount(runtime::ContactCase type)
        {
            switch (type)
            {
            case runtime::ContactCase::PP:
                return 2;
            case runtime::ContactCase::PE:
                return 3;
            case runtime::ContactCase::PT:
            case runtime::ContactCase::EE:
                return 4;
            }
            return 0;
        }

        ksk::engine::contact::EBarrierStencil barrierStencilType(
            runtime::ContactCase type)
        {
            switch (type)
            {
            case runtime::ContactCase::PP:
                return ksk::engine::contact::EBarrierStencil::PP;
            case runtime::ContactCase::PE:
                return ksk::engine::contact::EBarrierStencil::PE;
            case runtime::ContactCase::PT:
                return ksk::engine::contact::EBarrierStencil::PT;
            case runtime::ContactCase::EE:
                return ksk::engine::contact::EBarrierStencil::EE;
            }
            return ksk::engine::contact::EBarrierStencil::PP;
        }

        int findSampleIndex(const FEMSubsystem& subsystem, runtime::PointIdx point)
        {
            const std::vector<runtime::PointIdx>& points = subsystem.geometryPointIds();
            const auto it = std::find(points.begin(),
                                      points.end(),
                                      point);
            if (it == points.end())
            {
                return -1;
            }
            return static_cast<int>(it - points.begin());
        }

        glm::dvec3 localVertexPosition(const FEMSubsystem& subsystem,
                                       const FEMVertexSample& sample)
        {
            const TetMeshDesc& mesh =
                subsystem.meshes().at(static_cast<size_t>(sample.mesh));
            if (!mesh.initialPositions.empty())
            {
                return mesh.initialPositions.at(static_cast<size_t>(sample.vertex));
            }
            return mesh.vertices.at(static_cast<size_t>(sample.vertex));
        }

        glm::dvec3 localRestVertexPosition(const FEMSubsystem& subsystem,
                                           const FEMVertexSample& sample)
        {
            return subsystem.meshes()
                            .at(sample.mesh)
                            .vertices.at(sample.vertex);
        }

        struct LocalContactStencil
        {
            std::array<int, 4> qOffsets{-1, -1, -1, -1};
            std::array<glm::dvec3, 4> x{
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0)
            };
            std::array<glm::dvec3, 4> restX{
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0)
            };
            int count = 0;
            double dHat = 0.0;
            double surfaceOffset = 0.0;
            double kappa = 0.0;
        };

        std::optional<LocalContactStencil> gatherLocalContact(
            const FEMSubsystem& subsystem,
            const runtime::ContactStencil& contact)
        {
            LocalContactStencil local;
            local.count = contactPointCount(contact.type);
            local.dHat = contact.dHat;
            local.surfaceOffset = contact.thickness;
            local.kappa = contact.stiffness;

            if (local.count <= 0 || local.dHat <= 0.0 || local.kappa <= 0.0)
            {
                return std::nullopt;
            }

            for (int i = 0; i < local.count; ++i)
            {
                const int sample_index =
                    findSampleIndex(subsystem, contact.geometryIds[static_cast<size_t>(i)]);
                if (sample_index < 0)
                {
                    return std::nullopt;
                }
                const FEMVertexSample& sample = subsystem.samples().at(sample_index);
                local.qOffsets[i] = sample.qOffset;
                local.x[i] = localVertexPosition(subsystem, sample);
                local.restX[i] = localRestVertexPosition(subsystem, sample);
            }
            return local;
        }

        ksk::engine::contact::GIPCBarrierStencil makeBarrierStencil(
            const LocalContactStencil& contact,
            runtime::ContactCase type)
        {
            ksk::engine::contact::GIPCBarrierStencil barrier;
            barrier.type = barrierStencilType(type);
            barrier.x = contact.x;
            barrier.restX = contact.restX;
            barrier.dHat = contact.dHat;
            barrier.reservedDist = contact.surfaceOffset;
            barrier.kappa = contact.kappa;
            return barrier;
        }

        void scatterBarrierValues(
            const LocalContactStencil& contact,
            const std::array<glm::dvec3, 4>& values,
            runtime::DofView y)
        {
            for (int i = 0; i < contact.count; i++)
            {
                const int offset = contact.qOffsets[i];
                y[offset + 0] += values[i].x;
                y[offset + 1] += values[i].y;
                y[offset + 2] += values[i].z;
            }
        }

        std::array<glm::dvec3, 4> gatherBarrierDirection(
            const LocalContactStencil& contact,
            runtime::ConstDofView dq)
        {
            std::array<glm::dvec3, 4> direction{
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0)
            };
            for (int i = 0; i < contact.count; i++)
            {
                const int offset = contact.qOffsets[i];
                direction[i] = glm::dvec3(
                    dq[offset + 0], dq[offset + 1], dq[offset + 2]);
            }
            return direction;
        }

        double appendContactGradientAndEnergy(const FEMSubsystem& subsystem,
                                              const runtime::ContactStencil& contact,
                                              runtime::DofView* g)
        {
            const std::optional<LocalContactStencil> local =
                gatherLocalContact(subsystem, contact);
            if (!local)
            {
                return 0.0;
            }

            const ksk::engine::contact::GIPCBarrierStencil barrier =
                makeBarrierStencil(*local, contact.type);
            if (g == nullptr)
            {
                return ksk::engine::contact::computeGIPCBarrierEnergy(barrier);
            }
            const ksk::engine::contact::LocalBarrierGradient barrier_gradient =
                ksk::engine::contact::computeGIPCBarrierGradient(barrier);
            scatterBarrierValues(*local, barrier_gradient.gradient, *g);
            return barrier_gradient.energy;
        }

        void applyContactHessianProduct(const FEMSubsystem& subsystem,
                                        const runtime::ContactStencil& contact,
                                        runtime::ConstDofView dq,
                                        runtime::DofView y)
        {
            const std::optional<LocalContactStencil> local = gatherLocalContact(subsystem, contact);
            if (!local)
            {
                return;
            }

            const std::array<glm::dvec3, 4> product =
                ksk::engine::contact::computeGIPCBarrierHessianProduct(
                    makeBarrierStencil(*local, contact.type),
                    gatherBarrierDirection(*local, dq));
            scatterBarrierValues(*local, product, y);
        }

        void appendContactHessianTriplets(
            const FEMSubsystem& subsystem,
            const runtime::ContactStencil& contact,
            std::vector<Eigen::Triplet<double>>& triplets)
        {
            const std::optional<LocalContactStencil> local =
                gatherLocalContact(subsystem, contact);
            if (!local)
            {
                return;
            }

            const ksk::engine::contact::LocalBarrierHessian hessian =
                ksk::engine::contact::computeGIPCBarrierHessian(
                    makeBarrierStencil(*local, contact.type));
            for (int row_point = 0; row_point < local->count; row_point++)
            {
                const int row_offset = local->qOffsets[row_point];
                for (int col_point = 0; col_point < local->count; col_point++)
                {
                    const int col_offset = local->qOffsets[col_point];
                    const glm::dmat3& block =
                        hessian.blocks[row_point][col_point];
                    for (int row = 0; row < 3; row++)
                    {
                        for (int col = 0; col < 3; col++)
                        {
                            const double value = block[col][row];
                            if (value != 0.0)
                            {
                                triplets.emplace_back(row_offset + row,
                                                      col_offset + col,
                                                      value);
                            }
                        }
                    }
                }
            }
        }

        double gravityPotential(const Eigen::VectorXd& localQ,
                                const Eigen::VectorXd& massDiagonal,
                                const glm::dvec3& gravity)
        {
            double energy = 0.0;
            for (int vertex = 0; 3 * vertex + 2 < localQ.size(); ++vertex)
            {
                const int offset = 3 * vertex;
                const double mass = massDiagonal[offset];
                energy -= mass *
                    (gravity.x * localQ[offset + 0] +
                     gravity.y * localQ[offset + 1] +
                     gravity.z * localQ[offset + 2]);
            }
            return energy;
        }

        void addGravityGradient(Eigen::VectorXd& gradient,
                                const Eigen::VectorXd& massDiagonal,
                                const glm::dvec3& gravity)
        {
            for (int vertex = 0; 3 * vertex + 2 < gradient.size(); ++vertex)
            {
                const int offset = 3 * vertex;
                const double mass = massDiagonal[offset];
                gradient[offset + 0] -= mass * gravity.x;
                gradient[offset + 1] -= mass * gravity.y;
                gradient[offset + 2] -= mass * gravity.z;
            }
        }
    } // namespace

    FEMCPUBackend::FEMCPUBackend(FEMSubsystem& subsystem)
        : subsystem_(subsystem)
          , mass_diagonal_(buildMassDiagonal())
    {
    }

    void FEMCPUBackend::writeState(runtime::DofView q,
                                   runtime::DofView qdot) const
    {
        requireCPU(q, "FEMCPUBackend::writeState");
        requireCPU(qdot, "FEMCPUBackend::writeState");
        requireDofCount(q, subsystem_.range_.scalarCount, "FEMCPUBackend::writeState");
        requireDofCount(qdot, subsystem_.range_.scalarCount, "FEMCPUBackend::writeState");

        for (int mesh_index = 0;
             mesh_index < subsystem_.meshes_.size();
             ++mesh_index)
        {
            const TetMeshDesc& mesh = subsystem_.meshes_[mesh_index];
            const int base = subsystem_.mesh_offsets_[mesh_index].q;
            for (int vertex = 0; vertex < static_cast<int>(mesh.vertices.size());
                 ++vertex)
            {
                const glm::dvec3 position =
                    subsystem_.vertexPosition(mesh_index, vertex);
                const glm::dvec3 velocity =
                    subsystem_.vertexVelocity(mesh_index, vertex);
                q[base + 3 * vertex + 0] = position.x;
                q[base + 3 * vertex + 1] = position.y;
                q[base + 3 * vertex + 2] = position.z;
                qdot[base + 3 * vertex + 0] = velocity.x;
                qdot[base + 3 * vertex + 1] = velocity.y;
                qdot[base + 3 * vertex + 2] = velocity.z;
            }
        }
    }

    void FEMCPUBackend::readState(runtime::ConstDofView q,
                                  runtime::ConstDofView qdot)
    {
        requireCPU(q, "FEMCPUBackend::readState");
        requireCPU(qdot, "FEMCPUBackend::readState");
        requireDofCount(q, subsystem_.range_.scalarCount, "FEMCPUBackend::readState");
        requireDofCount(qdot, subsystem_.range_.scalarCount, "FEMCPUBackend::readState");

        for (int mesh_index = 0;
             mesh_index < static_cast<int>(subsystem_.meshes_.size());
             ++mesh_index)
        {
            TetMeshDesc& mesh = subsystem_.meshes_[mesh_index];
            const int base = subsystem_.mesh_offsets_[mesh_index].q;
            if (mesh.initialPositions.empty())
            {
                mesh.initialPositions = mesh.vertices;
            }
            if (mesh.initialVelocities.empty())
            {
                mesh.initialVelocities.resize(mesh.vertices.size(), glm::dvec3(0.0));
            }
            for (int vertex = 0; vertex < mesh.vertices.size(); vertex++)
            {
                subsystem_.setVertexPosition(
                    mesh_index, vertex,
                    glm::dvec3(
                        q[base + 3 * vertex + 0],
                        q[base + 3 * vertex + 1],
                        q[base + 3 * vertex + 2]));
                subsystem_.setVertexVelocity(
                    mesh_index, vertex,
                    glm::dvec3(qdot[base + 3 * vertex + 0],
                               qdot[base + 3 * vertex + 1],
                               qdot[base + 3 * vertex + 2]));
            }
        }
    }

    void FEMCPUBackend::beginStep(runtime::ConstDofView q,
                                  runtime::ConstDofView qdot,
                                  double dt)
    {
        step_dt_ = dt;
        step_start_ = gatherLocalVector(q, subsystem_.range_.scalarCount);
        inertial_target_ =
            step_start_ + dt * gatherLocalVector(qdot, subsystem_.range_.scalarCount);
        mass_diagonal_ = buildMassDiagonal();
    }

    void FEMCPUBackend::acceptStep(runtime::ConstDofView q,
                                   runtime::DofView qdot,
                                   double dt)
    {
        if (step_start_.size() != subsystem_.range_.scalarCount)
        {
            throw std::runtime_error("FEM CPU backend step has not begun");
        }

        const Eigen::VectorXd local_q = gatherLocalVector(q, subsystem_.range_.scalarCount);
        const Eigen::VectorXd local_qdot = (local_q - step_start_) / dt;
        requireDofCount(qdot, subsystem_.range_.scalarCount, "FEMCPUBackend::acceptStep");
        for (int i = 0; i < subsystem_.range_.scalarCount; ++i)
        {
            qdot[i] = local_qdot[i];
        }
    }

    double FEMCPUBackend::evaluateObjective(runtime::ConstDofView q,
                                            runtime::ConstDofView qdot,
                                            double dt)
    {
        (void)qdot;
        const Eigen::VectorXd local_q = gatherLocalVector(q, subsystem_.range_.scalarCount);
        const Eigen::VectorXd residual = local_q - inertial_target_;
        double objective =
            0.5 *
            (mass_diagonal_.array() * residual.array().square()).sum() /
            (dt * dt);
        objective += subsystem_.elasticEnergy(local_q);
        objective += gravityPotential(
            local_q, mass_diagonal_, subsystem_.gravity_);

        for (const FEMSubsystem::ActiveConstraint& constraint :
             subsystem_.active_constraints_)
        {
            const double value = local_q[constraint.qOffset];
            const double diff = value - constraint.target;
            objective += 0.5 * constraint.stiffness * diff * diff;
        }
        for (const runtime::ContactStencil& contact : subsystem_.internal_contacts_)
        {
            objective += appendContactGradientAndEnergy(subsystem_, contact, nullptr);
        }
        return objective;
    }

    void FEMCPUBackend::updateInternalConstraints(double time, double dt)
    {
        subsystem_.updateConstraintTargets(time);
    }

    void FEMCPUBackend::prepareLocalOperator(double dt)
    {
        if (mass_diagonal_.size() != subsystem_.range_.scalarCount)
        {
            mass_diagonal_ = buildMassDiagonal();
        }

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(static_cast<size_t>(
            subsystem_.range_.scalarCount +
            144 +
            144 * subsystem_.internal_contacts_.size()));
        for (int i = 0; i < subsystem_.range_.scalarCount; ++i)
        {
            triplets.emplace_back(i, i, mass_diagonal_[i] / (dt * dt));
        }
        for (const FEMSubsystem::ActiveConstraint& constraint :
             subsystem_.active_constraints_)
        {
            triplets.emplace_back(
                constraint.qOffset, constraint.qOffset, constraint.stiffness);
        }
        subsystem_.assembleElasticHessian(subsystem_.gatherCurrentState(), triplets);
        for (const runtime::ContactStencil& contact : subsystem_.internal_contacts_)
        {
            appendContactHessianTriplets(subsystem_, contact, triplets);
        }

        local_matrix_.resize(subsystem_.range_.scalarCount,
                             subsystem_.range_.scalarCount);
        local_matrix_.setFromTriplets(triplets.begin(), triplets.end());
        local_matrix_.makeCompressed();
    }

    void FEMCPUBackend::assembleLocalGradient(runtime::DofView g) const
    {
        requireCPU(g, "FEMCPUBackend::assembleLocalGradient");
        requireDofCount(g, subsystem_.range_.scalarCount, "FEMCPUBackend::assembleLocalGradient");
        if (step_dt_ <= 0.0)
        {
            throw std::runtime_error("FEM CPU backend step has not begun");
        }

        const Eigen::VectorXd current = subsystem_.gatherCurrentState();
        Eigen::VectorXd local_gradient =
            (mass_diagonal_.array() * (current - inertial_target_).array()).matrix() /
            (step_dt_ * step_dt_);
        subsystem_.assembleElasticGradient(current, local_gradient);
        addGravityGradient(
            local_gradient, mass_diagonal_, subsystem_.gravity_);
        for (const FEMSubsystem::ActiveConstraint& constraint :
             subsystem_.active_constraints_)
        {
            local_gradient[constraint.qOffset] +=
                constraint.stiffness *
                (current[constraint.qOffset] - constraint.target);
        }
        for (int i = 0; i < subsystem_.range_.scalarCount; ++i)
        {
            g[i] += local_gradient[i];
        }
        for (const runtime::ContactStencil& contact : subsystem_.internal_contacts_)
        {
            appendContactGradientAndEnergy(subsystem_, contact, &g);
        }
    }

    void FEMCPUBackend::applyLocalMatrix(runtime::ConstDofView x,
                                         runtime::DofView y) const
    {
        if (local_matrix_.rows() != subsystem_.range_.scalarCount)
        {
            throw std::runtime_error("FEM CPU backend local operator is not prepared");
        }

        requireDofCount(x, subsystem_.range_.scalarCount, "FEMCPUBackend::applyLocalMatrix");
        requireDofCount(y, subsystem_.range_.scalarCount, "FEMCPUBackend::applyLocalMatrix");
        const Eigen::VectorXd local_x = gatherLocalVector(x, subsystem_.range_.scalarCount);
        const Eigen::VectorXd local_y = local_matrix_ * local_x;
        for (int i = 0; i < subsystem_.range_.scalarCount; ++i)
        {
            y[i] += local_y[i];
        }
    }

    void FEMCPUBackend::solveLocalSystem(runtime::ConstDofView b,
                                         runtime::DofView x) const
    {
        requireCPU(b, "FEMCPUBackend::solveLocalSystem");
        requireCPU(x, "FEMCPUBackend::solveLocalSystem");
        requireDofCount(b, subsystem_.range_.scalarCount, "FEMCPUBackend::solveLocalSystem");
        requireDofCount(x, subsystem_.range_.scalarCount, "FEMCPUBackend::solveLocalSystem");
        if (local_matrix_.rows() != subsystem_.range_.scalarCount)
        {
            throw std::runtime_error("FEM CPU backend local operator is not prepared");
        }

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(local_matrix_);
        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("failed to factor FEM CPU backend local matrix");
        }
        const Eigen::VectorXd local_x =
            solver.solve(gatherLocalVector(b, subsystem_.range_.scalarCount));
        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("failed to solve FEM CPU backend local system");
        }

        for (int i = 0; i < subsystem_.range_.scalarCount; ++i)
        {
            x[i] = local_x[i];
        }
    }

    void FEMCPUBackend::mapLocalDirectionToGeometry(runtime::ConstDofView localDq,
                                               runtime::GeometryView globalDx) const
    {
        requireCPU(localDq, "FEMCPUBackend::mapLocalDirectionToGeometry");
        requireCPU(globalDx, "FEMCPUBackend::mapLocalDirectionToGeometry");
        if (subsystem_.geometry_points_.size() != subsystem_.samples_.size())
        {
            throw std::runtime_error("FEM geometry point mapping is stale");
        }

        for (int sample_index = 0;
             sample_index < static_cast<int>(subsystem_.samples_.size());
             ++sample_index)
        {
            const runtime::PointIdx point =
                subsystem_.geometry_points_[sample_index];
            if (point < 0 || point >= globalDx.pointCount())
            {
                throw std::invalid_argument(
                    "FEM global geometry direction buffer is too small");
            }
            const int offset = subsystem_.samples_[sample_index].qOffset;
            globalDx[point] =
                glm::dvec3(localDq[offset + 0],
                           localDq[offset + 1],
                           localDq[offset + 2]);
        }
    }

    void FEMCPUBackend::scatterContactGradient(
        std::span<const runtime::PointIdx> points,
        runtime::ConstGeometryView pointGradient,
        runtime::DofView g) const
    {
        requireCPU(pointGradient, "FEMCPUBackend::scatterContactGradient");
        requireCPU(g, "FEMCPUBackend::scatterContactGradient");
        if (points.size() != static_cast<size_t>(pointGradient.pointCount()))
        {
            throw std::invalid_argument("contact point and gradient counts differ");
        }
        requireDofCount(g, subsystem_.range_.scalarCount, "FEMCPUBackend::scatterContactGradient");

        for (size_t i = 0; i < points.size(); ++i)
        {
            const auto it = std::find(subsystem_.geometry_points_.begin(),
                                      subsystem_.geometry_points_.end(),
                                      points[i]);
            if (it == subsystem_.geometry_points_.end())
            {
                continue;
            }
            const int sample_index =
                static_cast<int>(it - subsystem_.geometry_points_.begin());
            const int offset = subsystem_.samples_[sample_index].qOffset;
            g[offset + 0] += pointGradient[static_cast<int>(i)].x;
            g[offset + 1] += pointGradient[static_cast<int>(i)].y;
            g[offset + 2] += pointGradient[static_cast<int>(i)].z;
        }
    }

    void FEMCPUBackend::applyContactGeometryHessianProduct(
        std::span<const runtime::PointIdx> gradientPoints,
        runtime::ConstGeometryView pointGradient,
        std::span<const runtime::PointIdx> productPoints,
        runtime::ConstGeometryView pointHessianProduct,
        runtime::ConstDofView localDq,
        runtime::DofView localY) const
    {
        requireCPU(pointGradient, "FEMCPUBackend::applyContactGeometryHessianProduct");
        requireCPU(pointHessianProduct, "FEMCPUBackend::applyContactGeometryHessianProduct");
        requireCPU(localDq, "FEMCPUBackend::applyContactGeometryHessianProduct");
        requireCPU(localY, "FEMCPUBackend::applyContactGeometryHessianProduct");
        if (gradientPoints.size() != static_cast<size_t>(pointGradient.pointCount()))
        {
            throw std::invalid_argument("contact point and gradient counts differ");
        }
        requireDofCount(localDq,
                        subsystem_.range_.scalarCount,
                        "FEMCPUBackend::applyContactGeometryHessianProduct");

        (void)gradientPoints;
        (void)localDq;

        scatterContactGradient(productPoints, pointHessianProduct, localY);
    }

    void FEMCPUBackend::applyInternalContactHessian(
        runtime::ConstDofView localDq,
        runtime::DofView localY) const
    {
        requireCPU(localDq, "FEMCPUBackend::applyInternalContactHessian");
        requireCPU(localY, "FEMCPUBackend::applyInternalContactHessian");
        requireDofCount(localDq, subsystem_.range_.scalarCount,
                        "FEMCPUBackend::applyInternalContactHessian");
        requireDofCount(localY, subsystem_.range_.scalarCount,
                        "FEMCPUBackend::applyInternalContactHessian");

        for (const runtime::ContactStencil& contact : subsystem_.internal_contacts_)
        {
            applyContactHessianProduct(subsystem_, contact, localDq, localY);
        }
    }

    Eigen::VectorXd FEMCPUBackend::buildMassDiagonal() const
    {
        Eigen::VectorXd mass =
            Eigen::VectorXd::Zero(subsystem_.range_.scalarCount);
        for (int mesh_index = 0;
             mesh_index < static_cast<int>(subsystem_.meshes_.size());
             ++mesh_index)
        {
            const TetMeshDesc& mesh = subsystem_.meshes_[mesh_index];
            const double vertex_mass = std::max(1.0e-12, mesh.material.density);
            const int base = subsystem_.mesh_offsets_[mesh_index].q;
            for (int vertex = 0; vertex < static_cast<int>(mesh.vertices.size());
                 ++vertex)
            {
                mass[base + 3 * vertex + 0] = vertex_mass;
                mass[base + 3 * vertex + 1] = vertex_mass;
                mass[base + 3 * vertex + 2] = vertex_mass;
            }
        }
        return mass;
    }
} // namespace ksk::engine::fem
