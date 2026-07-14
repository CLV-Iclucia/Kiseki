#include <DER/cpu/der-cpu-backend.h>

#include <DER/der-subsystem.h>

#include <Eigen/SparseCholesky>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace ksk::der
{
    namespace
    {
        static_assert(sizeof(RodBlock) == 4 * sizeof(double));

        void ensureWritable(Eigen::VectorXd& values, int minimumSize)
        {
            if (values.size() >= minimumSize)
            {
                return;
            }
            const Eigen::Index old_size = values.size();
            values.conservativeResize(minimumSize);
            values.segment(old_size, minimumSize - old_size).setZero();
        }

        void requireCPU(const runtime::DofBuffer& buffer, const char* operation)
        {
            if (!buffer.isCPU())
            {
                throw std::runtime_error(std::string(operation) +
                    " requires a CPU DofBuffer");
            }
        }

        void requireCPU(const runtime::GeometryBuffer& buffer, const char* operation)
        {
            if (!buffer.isCPU())
            {
                throw std::runtime_error(std::string(operation) +
                    " requires a CPU GeometryBuffer");
            }
        }
    } // namespace

    DERCPUBackend::DERCPUBackend(DERSubsystem& subsystem)
        : subsystem_(subsystem)
    {
    }

    void DERCPUBackend::writeState(runtime::DofBuffer& q,
                                   runtime::DofBuffer& qdot) const
    {
        requireCPU(q, "DERCpuBackend::writeState");
        requireCPU(qdot, "DERCpuBackend::writeState");

        const runtime::DofRange range = subsystem_.dofRange();
        Eigen::VectorXd& q_values = q.cpu();
        Eigen::VectorXd& qdot_values = qdot.cpu();
        ensureWritable(q_values, range.scalarOffset + range.scalarCount);
        ensureWritable(qdot_values, range.scalarOffset + range.scalarCount);

        const auto& rods = subsystem_.rods();
        const auto& offsets = subsystem_.rodOffsets();
        for (int rod_index = 0; rod_index < static_cast<int>(rods.size());
             ++rod_index)
        {
            const Rod& rod = rods[rod_index];
            const int base = range.scalarOffset + offsets[rod_index].q;
            for (int vertex = 0; vertex < static_cast<int>(rod.state().size());
                 ++vertex)
            {
                for (int lane = 0; lane < 4; ++lane)
                {
                    q_values[base + 4 * vertex + lane] =
                        rod.state().blocks[vertex][lane];
                    qdot_values[base + 4 * vertex + lane] =
                        rod.velocity().blocks[vertex][lane];
                }
            }
        }
    }

    void DERCPUBackend::readState(runtime::DofBuffer& q,
                                  runtime::DofBuffer& qdot)
    {
        requireCPU(q, "DERCpuBackend::readState");
        requireCPU(qdot, "DERCpuBackend::readState");

        Eigen::VectorXd& q_values = q.cpu();
        Eigen::VectorXd& qdot_values = qdot.cpu();
        const auto& offsets = subsystem_.rodOffsets();
        auto& rods = subsystem_.rods();
        const runtime::DofRange range = subsystem_.dofRange();
        ensureWritable(q_values, range.scalarOffset + range.scalarCount);
        ensureWritable(qdot_values, range.scalarOffset + range.scalarCount);

        for (int rod_index = 0; rod_index < static_cast<int>(rods.size());
             ++rod_index)
        {
            Rod& rod = rods[rod_index];
            const int base = offsets[rod_index].q;
            const int block_count = static_cast<int>(rod.state().size());
            auto* q_blocks = reinterpret_cast<RodBlock*>(
                q_values.data() + vectorOffset(q_values, base));
            auto* qdot_blocks = reinterpret_cast<RodBlock*>(
                qdot_values.data() + vectorOffset(qdot_values, base));
            rod.bindState(std::span<RodBlock>(q_blocks, block_count),
                          std::span<RodBlock>(qdot_blocks, block_count));
        }
    }

    void DERCPUBackend::beginStep(const runtime::DofBuffer& q,
                                  const runtime::DofBuffer& qdot,
                                  double dt)
    {
        requireCPU(q, "DERCpuBackend::beginStep");
        requireCPU(qdot, "DERCpuBackend::beginStep");
        step_dt_ = dt;
        step_start_ = gatherLocalVector(q.cpu());
        inertial_target_ = step_start_ + dt * gatherLocalVector(qdot.cpu());

        mass_diagonal_ = Eigen::VectorXd::Zero(subsystem_.localScalarCount());
        const auto& rods = subsystem_.rods();
        const auto& offsets = subsystem_.rodOffsets();
        step_previous_states_.clear();
        step_previous_states_.reserve(rods.size());
        for (int rod_index = 0; rod_index < static_cast<int>(rods.size());
             ++rod_index)
        {
            step_previous_states_.emplace_back(rods[rod_index].state().blocks.begin(),
                                               rods[rod_index].state().blocks.end());
            const Eigen::VectorXd rod_mass = rods[rod_index].massDiagonal();
            const int base = offsets[rod_index].q;
            mass_diagonal_.segment(base, rod_mass.size()) = rod_mass;
        }
    }

    void DERCPUBackend::acceptStep(const runtime::DofBuffer& q,
                                   runtime::DofBuffer& qdot,
                                   double dt)
    {
        requireCPU(q, "DERCpuBackend::acceptStep");
        requireCPU(qdot, "DERCpuBackend::acceptStep");
        if (step_start_.size() != subsystem_.localScalarCount())
        {
            throw std::runtime_error("DER CPU backend step has not begun");
        }

        const runtime::DofRange range = subsystem_.dofRange();
        Eigen::VectorXd& qdot_values = qdot.cpu();
        ensureWritable(qdot_values, range.scalarOffset + range.scalarCount);
        const Eigen::VectorXd local_q = gatherLocalVector(q.cpu());
        const Eigen::VectorXd local_qdot = (local_q - step_start_) / dt;
        for (int i = 0; i < range.scalarCount; ++i)
        {
            qdot_values[vectorOffset(qdot_values, i)] = local_qdot[i];
        }

        auto& rods = subsystem_.rods();
        for (int rod_index = 0; rod_index < static_cast<int>(rods.size());
             ++rod_index)
        {
            if (rod_index >= static_cast<int>(step_previous_states_.size()))
            {
                continue;
            }
            std::vector<RodBlock>& previous_blocks =
                step_previous_states_[rod_index];
            if (previous_blocks.size() != rods[rod_index].state().size())
            {
                continue;
            }
            RodState previous{
                std::span<RodBlock>(previous_blocks.data(), previous_blocks.size())
            };
            rods[rod_index].transportReferenceFrames(previous);
        }
    }

    double DERCPUBackend::evaluateObjective(const runtime::DofBuffer& q,
                                            const runtime::DofBuffer& qdot,
                                            double dt)
    {
        requireCPU(q, "DERCpuBackend::evaluateObjective");
        requireCPU(qdot, "DERCpuBackend::evaluateObjective");

        const Eigen::VectorXd local_q = gatherLocalVector(q.cpu());
        double objective = 0.0;
        const Eigen::VectorXd inertial_residual = local_q - inertial_target_;
        objective +=
            0.5 *
            (mass_diagonal_.array() * inertial_residual.array().square()).sum() /
            (dt * dt);

        const auto& rods = subsystem_.rods();
        for (int i = 0; i < rods.size(); i++)
        {
            const RodEvaluation evaluation = evaluateRod(rods[i], i);
            if (!evaluation.valid)
            {
                throw std::runtime_error(evaluation.diagnostic);
            }
            objective += evaluation.energy.total();
        }
        return objective;
    }

    void DERCPUBackend::updateInternalConstraints(double time, double dt)
    {

    }

    void DERCPUBackend::prepareLocalOperator(double dt)
    {
        const auto& rods = subsystem_.rods();
        const auto& offsets = subsystem_.rodOffsets();
        evaluations_.clear();
        evaluations_.reserve(rods.size());

        std::vector<Eigen::Triplet<double>> triplets;
        if (mass_diagonal_.size() != subsystem_.localScalarCount())
        {
            mass_diagonal_ = Eigen::VectorXd::Zero(subsystem_.localScalarCount());
            for (int rod_index = 0; rod_index < static_cast<int>(rods.size());
                 ++rod_index)
            {
                const Eigen::VectorXd rod_mass = rods[rod_index].massDiagonal();
                const int base = offsets[rod_index].q;
                mass_diagonal_.segment(base, rod_mass.size()) = rod_mass;
            }
        }
        for (int rod_index = 0; rod_index < static_cast<int>(rods.size());
             ++rod_index)
        {
            const Rod& rod = rods[rod_index];
            RodEvaluation evaluation = evaluateRod(rod, rod_index);
            if (!evaluation.valid)
            {
                throw std::runtime_error(evaluation.diagnostic);
            }

            const int base = offsets[rod_index].q;
            const Eigen::VectorXd mass = rod.massDiagonal();
            for (Eigen::Index i = 0; i < mass.size(); ++i)
            {
                triplets.emplace_back(base + static_cast<int>(i),
                                      base + static_cast<int>(i),
                                      mass[i] / (dt * dt));
            }
            for (int outer = 0; outer < evaluation.hessian.outerSize(); ++outer)
            {
                for (Eigen::SparseMatrix<double>::InnerIterator it(evaluation.hessian,
                                                                   outer);
                     it; ++it)
                {
                    triplets.emplace_back(base + static_cast<int>(it.row()),
                                          base + static_cast<int>(it.col()),
                                          it.value());
                }
            }

            evaluations_.push_back(std::move(evaluation));
        }

        local_matrix_.resize(subsystem_.localScalarCount(),
                             subsystem_.localScalarCount());
        local_matrix_.setFromTriplets(triplets.begin(), triplets.end());
        local_matrix_.makeCompressed();
    }

    void DERCPUBackend::assembleLocalGradient(runtime::DofBuffer& g) const
    {
        if (evaluations_.size() != subsystem_.rods().size())
        {
            throw std::runtime_error("DER CPU backend local operator is not prepared");
        }
        requireCPU(g, "DERCpuBackend::assembleLocalGradient");

        const runtime::DofRange range = subsystem_.dofRange();
        const auto& offsets = subsystem_.rodOffsets();
        Eigen::VectorXd& g_values = g.cpu();
        ensureWritable(g_values, range.scalarOffset + range.scalarCount);

        Eigen::VectorXd local_gradient = Eigen::VectorXd::Zero(range.scalarCount);
        if (inertial_target_.size() == range.scalarCount &&
            mass_diagonal_.size() == range.scalarCount)
        {
            local_gradient +=
                (mass_diagonal_.array() *
                    (gatherCurrentState() - inertial_target_).array()).matrix() /
                (step_dt_ * step_dt_);
        }

        for (int rod_index = 0; rod_index < static_cast<int>(subsystem_.rods().size());
             ++rod_index)
        {
            const int base = offsets[rod_index].q;
            const RodEvaluation& evaluation = evaluations_[rod_index];
            for (int vertex = 0; vertex < static_cast<int>(evaluation.gradient.size());
                 ++vertex)
            {
                for (int lane = 0; lane < 4; ++lane)
                {
                    local_gradient[base + 4 * vertex + lane] +=
                        evaluation.gradient[vertex][lane];
                }
            }
        }
        for (int i = 0; i < range.scalarCount; ++i)
        {
            addToVector(g_values, i, local_gradient[i]);
        }
    }

    void DERCPUBackend::applyLocalMatrix(const runtime::DofBuffer& x,
                                         runtime::DofBuffer& y) const
    {
        if (local_matrix_.rows() != subsystem_.localScalarCount())
        {
            throw std::runtime_error("DER CPU backend local matrix is not prepared");
        }
        requireCPU(x, "DERCpuBackend::applyLocalMatrix");
        requireCPU(y, "DERCpuBackend::applyLocalMatrix");

        const runtime::DofRange range = subsystem_.dofRange();
        const Eigen::VectorXd local_x = gatherLocalVector(x.cpu());
        const Eigen::VectorXd local_y = local_matrix_ * local_x;
        Eigen::VectorXd& y_values = y.cpu();
        ensureWritable(y_values, range.scalarOffset + range.scalarCount);
        for (int i = 0; i < range.scalarCount; ++i)
        {
            addToVector(y_values, i, local_y[i]);
        }
    }

    void DERCPUBackend::solveLocalSystem(const runtime::DofBuffer& b,
                                         runtime::DofBuffer& x) const
    {
        if (local_matrix_.rows() != subsystem_.localScalarCount())
        {
            throw std::runtime_error("DER CPU backend local matrix is not prepared");
        }
        requireCPU(b, "DERCpuBackend::solveLocalSystem");
        requireCPU(x, "DERCpuBackend::solveLocalSystem");

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(local_matrix_);
        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("failed to factor DER CPU backend local matrix");
        }
        const Eigen::VectorXd local_x = solver.solve(gatherLocalVector(b.cpu()));
        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("failed to solve DER CPU backend local system");
        }

        const runtime::DofRange range = subsystem_.dofRange();
        Eigen::VectorXd& x_values = x.cpu();
        ensureWritable(x_values, range.scalarOffset + range.scalarCount);
        for (int i = 0; i < range.scalarCount; ++i)
        {
            x_values[vectorOffset(x_values, i)] = local_x[i];
        }
    }

    void DERCPUBackend::mapDirectionToGeometry(const runtime::DofBuffer& dq,
                                               runtime::GeometryBuffer& dx) const
    {
        requireCPU(dq, "DERCpuBackend::mapDirectionToGeometry");
        requireCPU(dx, "DERCpuBackend::mapDirectionToGeometry");

        const Eigen::VectorXd& dq_values = dq.cpu();
        const auto& samples = subsystem_.geometrySamples();
        std::vector<glm::dvec3>& dx_values = dx.cpu();
        if (dx_values.size() < samples.size())
        {
            throw std::invalid_argument("DER geometry direction buffer is too small");
        }

        for (int sample = 0; sample < static_cast<int>(samples.size()); ++sample)
        {
            const int offset = samples[sample].qOffset;
            dx_values[sample] =
                glm::dvec3(dq_values[vectorOffset(dq_values, offset + 0)],
                           dq_values[vectorOffset(dq_values, offset + 1)],
                           dq_values[vectorOffset(dq_values, offset + 2)]);
        }
    }

    void DERCPUBackend::scatterContactGradient(
        std::span<const runtime::GeometryPointId> points,
        const runtime::GeometryBuffer& pointGradient,
        runtime::DofBuffer& g) const
    {
        requireCPU(pointGradient, "DERCpuBackend::scatterContactGradient");
        requireCPU(g, "DERCpuBackend::scatterContactGradient");

        const std::vector<glm::dvec3>& point_gradient = pointGradient.cpu();
        if (points.size() != point_gradient.size())
        {
            throw std::invalid_argument("contact point and gradient counts differ");
        }

        const runtime::DofRange range = subsystem_.dofRange();
        Eigen::VectorXd& g_values = g.cpu();
        ensureWritable(g_values, range.scalarOffset + range.scalarCount);

        const auto& geometry_points = subsystem_.geometryPointIds();
        const auto& samples = subsystem_.geometrySamples();
        for (size_t i = 0; i < points.size(); ++i)
        {
            const auto it =
                std::find(geometry_points.begin(), geometry_points.end(), points[i]);
            if (it == geometry_points.end())
            {
                continue;
            }
            const int sample = static_cast<int>(it - geometry_points.begin());
            const int offset = samples[sample].qOffset;
            addToVector(g_values, offset + 0, point_gradient[i].x);
            addToVector(g_values, offset + 1, point_gradient[i].y);
            addToVector(g_values, offset + 2, point_gradient[i].z);
        }
    }

    void DERCPUBackend::applyContactHessian(const runtime::DofBuffer& dq,
                                            const runtime::ContactTable& contacts,
                                            runtime::DofBuffer& y) const
    {
    }

    const RodEvaluation& DERCPUBackend::cachedEvaluation(int rod) const
    {
        return evaluations_.at(static_cast<size_t>(rod));
    }

    int DERCPUBackend::vectorOffset(const Eigen::VectorXd& values,
                                    int localOffset) const
    {
        const runtime::DofRange range = subsystem_.dofRange();
        if (values.size() == range.scalarCount)
        {
            return localOffset;
        }
        return range.scalarOffset + localOffset;
    }

    Eigen::VectorXd DERCPUBackend::gatherLocalVector(
        const Eigen::VectorXd& values) const
    {
        Eigen::VectorXd local(subsystem_.localScalarCount());
        for (int i = 0; i < subsystem_.localScalarCount(); ++i)
        {
            local[i] = values[vectorOffset(values, i)];
        }
        return local;
    }

    Eigen::VectorXd DERCPUBackend::gatherCurrentState() const
    {
        Eigen::VectorXd local = Eigen::VectorXd::Zero(subsystem_.localScalarCount());
        const auto& rods = subsystem_.rods();
        const auto& offsets = subsystem_.rodOffsets();
        for (int rod_index = 0; rod_index < static_cast<int>(rods.size());
             ++rod_index)
        {
            const Rod& rod = rods[rod_index];
            const int base = offsets[rod_index].q;
            for (int vertex = 0; vertex < static_cast<int>(rod.state().size());
                 ++vertex)
            {
                for (int lane = 0; lane < 4; ++lane)
                {
                    local[base + 4 * vertex + lane] = rod.state().blocks[vertex][lane];
                }
            }
        }
        return local;
    }

    RodEvaluation DERCPUBackend::evaluateRod(const Rod& rod, int rodIndex)
    {
        if (rodIndex >= 0 &&
            rodIndex < static_cast<int>(step_previous_states_.size()))
        {
            std::vector<RodBlock>& previous_blocks =
                step_previous_states_[rodIndex];
            if (previous_blocks.size() == rod.state().size())
            {
                RodState previous{
                    std::span<RodBlock>(previous_blocks.data(), previous_blocks.size())
                };
                return rod.evaluate(subsystem_.gravity(), previous);
            }
        }
        return rod.evaluate(subsystem_.gravity());
    }

    void DERCPUBackend::addToVector(Eigen::VectorXd& values,
                                    int localOffset,
                                    double value) const
    {
        values[vectorOffset(values, localOffset)] += value;
    }
} // namespace ksk::der
