#include <Runtime/contact-detection.h>

#include "../contact-detection-impl.h"

#include <Spatify/lbvh.h>

#include <array>
#include <algorithm>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <glm/geometric.hpp>
#include <tbb/parallel_for.h>

namespace ksk::runtime
{
    namespace
    {
        constexpr int kMaxACCDIterations = 2000;
        constexpr Real kACCDGapFraction = 0.1;
        constexpr Real kBVHMinimumExtent = 1.0e-12;

        enum class PointTriangleDistanceType
        {
            P_A,
            P_B,
            P_C,
            P_AB,
            P_BC,
            P_CA,
            P_ABC,
        };

        enum class EdgeEdgeDistanceType
        {
            A_C,
            A_D,
            B_C,
            B_D,
            AB_C,
            AB_D,
            A_CD,
            B_CD,
            AB_CD,
        };

        struct PrimitiveRef
        {
            int id = -1;
            Real reserved_dist = 0.0;
            spatify::BBox<Real, 3> bounds;
        };

        struct PrimitiveAccessor
        {
            using CoordType = Real;

            const std::vector<PrimitiveRef>* primitives = nullptr;

            [[nodiscard]] spatify::BBox<Real, 3> bbox(int index) const
            {
                return primitives->at(static_cast<size_t>(index)).bounds;
            }

            [[nodiscard]] int size() const
            {
                return static_cast<int>(primitives->size());
            }
        };

        spatify::BBox<Real, 3> ensureFiniteBVHBounds(spatify::BBox<Real, 3> bounds)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                const Real extent = bounds.hi[axis] - bounds.lo[axis];
                if (extent >= kBVHMinimumExtent)
                {
                    continue;
                }
                const Real centre = (bounds.lo[axis] + bounds.hi[axis]) * 0.5;
                bounds.lo[axis] = centre - kBVHMinimumExtent;
                bounds.hi[axis] = centre + kBVHMinimumExtent;
            }
            return bounds;
        }

        void makeBoundsSafeForLBVH(std::vector<PrimitiveRef>& primitives)
        {
            for (PrimitiveRef& primitive : primitives)
            {
                primitive.bounds = ensureFiniteBVHBounds(primitive.bounds);
            }
        }

        [[nodiscard]] spatify::LBVH<Real> buildLBVH(
            std::vector<PrimitiveRef>& primitives)
        {
            makeBoundsSafeForLBVH(primitives);
            spatify::LBVH<Real> tree;
            if (!primitives.empty())
            {
                tree.update(PrimitiveAccessor{.primitives = &primitives});
            }
            return tree;
        }

        void sortUniquePairs(std::vector<std::array<int, 2>>& pairs)
        {
            std::ranges::sort(pairs);
            pairs.erase(std::ranges::unique(pairs).begin(), pairs.end());
        }

        void finalizeCollisionWorkList(ContactWorkList& work)
        {
            sortUniquePairs(work.deformablePointTriangles);
            sortUniquePairs(work.deformableEdgeEdges);
        }

        ContactStencil makeStencil(ContactCase type,
                                   std::array<int, 4> ids,
                                   const ContactDetectionConfig& config,
                                   Real reserved_dist = 0.0)
        {
            return ContactStencil{
                .type = type,
                .geometryIds = ids,
                .dHat = config.dHat,
                .stiffness = config.stiffness,
                .thickness = reserved_dist,
            };
        }

        ContactCandidate makeCandidate(EContactCandidate kind,
                                       std::array<int, 4> ids,
                                       const ContactDetectionConfig& config,
                                       Real reserved_distance = 0.0)
        {
            return ContactCandidate{
                .kind = kind,
                .geometryIds = ids,
                .detectionDistance = config.detectionDistance,
                .dHat = config.dHat,
                .stiffness = config.stiffness,
                .reservedDistance = reserved_distance,
            };
        }

        Real pointRadius(const GlobalGeometryManager& geometry, PointIdx point)
        {
            return geometry.points.at(static_cast<size_t>(point)).radius;
        }

        Real triangleSurfaceOffset(const GlobalGeometryManager& geometry,
                                   const std::array<PointIdx, 3>& triangle)
        {
            Real offset = 0.0;
            for (const GeometryTriangle& candidate : geometry.triangles)
            {
                const std::array points{candidate.p0, candidate.p1, candidate.p2};
                const bool same =
                    (points[0] == triangle[0] || points[0] == triangle[1] ||
                        points[0] == triangle[2]) &&
                    (points[1] == triangle[0] || points[1] == triangle[1] ||
                        points[1] == triangle[2]) &&
                    (points[2] == triangle[0] || points[2] == triangle[1] ||
                        points[2] == triangle[2]);
                if (same)
                {
                    offset = std::max(offset, candidate.thickness);
                }
            }
            return offset;
        }

        Real edgeSurfaceOffset(const GlobalGeometryManager& geometry,
                               PointIdx p0,
                               PointIdx p1,
                               Real fallback = 0.0)
        {
            Real offset = std::max({
                pointRadius(geometry, p0),
                pointRadius(geometry, p1),
                fallback
            });
            for (const GeometryEdge& edge : geometry.edges)
            {
                if ((edge.p0 == p0 && edge.p1 == p1) ||
                    (edge.p0 == p1 && edge.p1 == p0))
                {
                    offset = std::max(offset, edge.radius);
                }
            }
            return offset;
        }

        Real pointEdgeSurfaceOffset(const GlobalGeometryManager& geometry,
                                    PointIdx point,
                                    PointIdx edge0,
                                    PointIdx edge1,
                                    Real fallback = 0.0)
        {
            return pointRadius(geometry, point) +
                edgeSurfaceOffset(geometry, edge0, edge1, fallback);
        }

        Real distanceSqrPointPoint(const glm::dvec3& a, const glm::dvec3& b)
        {
            const glm::dvec3 d = a - b;
            return glm::dot(d, d);
        }

        Real distanceSqrPointLine(const glm::dvec3& p,
                                  const glm::dvec3& a,
                                  const glm::dvec3& b)
        {
            const glm::dvec3 e = b - a;
            const Real e_len2 = glm::dot(e, e);
            if (e_len2 == 0.0)
            {
                return distanceSqrPointPoint(p, a);
            }
            const glm::dvec3 c = glm::cross(e, p - a);
            return glm::dot(c, c) / e_len2;
        }

        Real distanceSqrPointPlane(const glm::dvec3& p,
                                   const glm::dvec3& a,
                                   const glm::dvec3& b,
                                   const glm::dvec3& c)
        {
            const glm::dvec3 normal = glm::cross(b - a, c - a);
            const Real n_len2 = glm::dot(normal, normal);
            if (n_len2 == 0.0)
            {
                return std::min({
                    distanceSqrPointLine(p, a, b),
                    distanceSqrPointLine(p, b, c),
                    distanceSqrPointLine(p, c, a)
                });
            }
            const Real d = glm::dot(p - a, normal);
            return d * d / n_len2;
        }

        Real distanceSqrLineLine(const glm::dvec3& a0,
                                 const glm::dvec3& a1,
                                 const glm::dvec3& b0,
                                 const glm::dvec3& b1)
        {
            const glm::dvec3 normal = glm::cross(a1 - a0, b1 - b0);
            const Real n_len2 = glm::dot(normal, normal);
            if (n_len2 == 0.0)
            {
                return distanceSqrPointLine(a0, b0, b1);
            }
            const Real d = glm::dot(a0 - b0, normal);
            return d * d / n_len2;
        }

        void pointTriangleEdgeTest(const glm::dvec3& from,
                                   const glm::dvec3& to,
                                   const glm::dvec3& point,
                                   const glm::dvec3& normal,
                                   Real& s,
                                   Real& t)
        {
            const glm::dvec3 e = to - from;
            const glm::dvec3 n = glm::cross(e, normal);
            const Real a00 = glm::dot(e, e);
            const Real a01 = glm::dot(e, n);
            const Real a11 = glm::dot(n, n);
            const Real b0 = glm::dot(e, point - from);
            const Real b1 = glm::dot(n, point - from);
            const Real det = a00 * a11 - a01 * a01;
            if (det == 0.0)
            {
                s = 0.0;
                t = -1.0;
                return;
            }
            s = (a11 * b0 - a01 * b1) / det;
            t = (a00 * b1 - a01 * b0) / det;
        }

        PointTriangleDistanceType decidePointTriangleDistanceType(
            const glm::dvec3& p,
            const glm::dvec3& a,
            const glm::dvec3& b,
            const glm::dvec3& c)
        {
            const glm::dvec3 normal = glm::cross(b - a, c - a);
            if (glm::dot(normal, normal) == 0.0)
            {
                const Real da = distanceSqrPointLine(p, a, b);
                const Real db = distanceSqrPointLine(p, b, c);
                const Real dc = distanceSqrPointLine(p, c, a);
                if (da <= db && da <= dc)
                {
                    return PointTriangleDistanceType::P_AB;
                }
                if (db <= dc)
                {
                    return PointTriangleDistanceType::P_BC;
                }
                return PointTriangleDistanceType::P_CA;
            }

            Real s0 = 0.0;
            Real t0 = 0.0;
            pointTriangleEdgeTest(a, b, p, normal, s0, t0);
            if (s0 > 0.0 && s0 < 1.0 && t0 >= 0.0)
            {
                return PointTriangleDistanceType::P_AB;
            }

            Real s1 = 0.0;
            Real t1 = 0.0;
            pointTriangleEdgeTest(b, c, p, normal, s1, t1);
            if (s1 > 0.0 && s1 < 1.0 && t1 >= 0.0)
            {
                return PointTriangleDistanceType::P_BC;
            }

            Real s2 = 0.0;
            Real t2 = 0.0;
            pointTriangleEdgeTest(c, a, p, normal, s2, t2);
            if (s2 > 0.0 && s2 < 1.0 && t2 >= 0.0)
            {
                return PointTriangleDistanceType::P_CA;
            }

            if (s0 <= 0.0 && s2 >= 1.0)
            {
                return PointTriangleDistanceType::P_A;
            }
            if (s1 <= 0.0 && s0 >= 1.0)
            {
                return PointTriangleDistanceType::P_B;
            }
            if (s2 <= 0.0 && s1 >= 1.0)
            {
                return PointTriangleDistanceType::P_C;
            }
            return PointTriangleDistanceType::P_ABC;
        }

        EdgeEdgeDistanceType decideEdgeEdgeParallelDistanceType(
            const glm::dvec3& ea0,
            const glm::dvec3& ea1,
            const glm::dvec3& eb0,
            const glm::dvec3& eb1)
        {
            const glm::dvec3 ea = ea1 - ea0;
            const Real ea_sqr = glm::dot(ea, ea);
            if (ea_sqr == 0.0)
            {
                return EdgeEdgeDistanceType::A_C;
            }
            const Real alpha = glm::dot(eb0 - ea0, ea) / ea_sqr;
            const Real beta = glm::dot(eb1 - ea0, ea) / ea_sqr;

            int eac = 0;
            int ebc = 0;
            if (alpha < 0.0)
            {
                eac = (0.0 <= beta && beta <= 1.0) ? 2 : 0;
                ebc = (beta <= alpha) ? 0 : (beta <= 1.0 ? 1 : 2);
            }
            else if (alpha > 1.0)
            {
                eac = (0.0 <= beta && beta <= 1.0) ? 2 : 1;
                ebc = (beta >= alpha) ? 0 : (0.0 <= beta ? 1 : 2);
            }
            else
            {
                eac = 2;
                ebc = 0;
            }

            if (ebc < 2)
            {
                const int code = (eac << 1) | ebc;
                switch (code)
                {
                case 0:
                    return EdgeEdgeDistanceType::A_C;
                case 1:
                    return EdgeEdgeDistanceType::A_D;
                case 2:
                    return EdgeEdgeDistanceType::B_C;
                case 3:
                    return EdgeEdgeDistanceType::B_D;
                case 4:
                    return EdgeEdgeDistanceType::AB_C;
                default:
                    return EdgeEdgeDistanceType::AB_D;
                }
            }

            return eac == 0
                       ? EdgeEdgeDistanceType::A_CD
                       : EdgeEdgeDistanceType::B_CD;
        }

        EdgeEdgeDistanceType decideEdgeEdgeDistanceType(const glm::dvec3& ea0,
                                                        const glm::dvec3& ea1,
                                                        const glm::dvec3& eb0,
                                                        const glm::dvec3& eb1)
        {
            constexpr Real parallel_threshold = 1.0e-20;
            const glm::dvec3 u = ea1 - ea0;
            const glm::dvec3 v = eb1 - eb0;
            const glm::dvec3 w = ea0 - eb0;

            const Real a = glm::dot(u, u);
            const Real b = glm::dot(u, v);
            const Real c = glm::dot(v, v);
            const Real d = glm::dot(u, w);
            const Real e = glm::dot(v, w);
            const Real det = a * c - b * b;

            if (a == 0.0 && c == 0.0)
            {
                return EdgeEdgeDistanceType::A_C;
            }
            if (a == 0.0)
            {
                return EdgeEdgeDistanceType::A_CD;
            }
            if (c == 0.0)
            {
                return EdgeEdgeDistanceType::AB_C;
            }

            const Real parallel_tolerance = parallel_threshold * std::max(1.0, a * c);
            const glm::dvec3 uxv = glm::cross(u, v);
            if (glm::dot(uxv, uxv) < parallel_tolerance)
            {
                return decideEdgeEdgeParallelDistanceType(ea0, ea1, eb0, eb1);
            }

            EdgeEdgeDistanceType default_case = EdgeEdgeDistanceType::AB_CD;
            const Real sN = b * e - c * d;
            Real tN = 0.0;
            Real tD = 0.0;
            if (sN <= 0.0)
            {
                tN = e;
                tD = c;
                default_case = EdgeEdgeDistanceType::A_CD;
            }
            else if (sN >= det)
            {
                tN = e + b;
                tD = c;
                default_case = EdgeEdgeDistanceType::B_CD;
            }
            else
            {
                tN = a * e - b * d;
                tD = det;
            }

            if (tN <= 0.0)
            {
                if (-d <= 0.0)
                {
                    return EdgeEdgeDistanceType::A_C;
                }
                return -d >= a
                           ? EdgeEdgeDistanceType::B_C
                           : EdgeEdgeDistanceType::AB_C;
            }
            if (tN >= tD)
            {
                if ((-d + b) <= 0.0)
                {
                    return EdgeEdgeDistanceType::A_D;
                }
                return (-d + b) >= a
                           ? EdgeEdgeDistanceType::B_D
                           : EdgeEdgeDistanceType::AB_D;
            }
            return default_case;
        }

        Real pointTriangleDistanceSqrByType(PointTriangleDistanceType type,
                                            const glm::dvec3& p,
                                            const glm::dvec3& a,
                                            const glm::dvec3& b,
                                            const glm::dvec3& c)
        {
            switch (type)
            {
            case PointTriangleDistanceType::P_A:
                return distanceSqrPointPoint(p, a);
            case PointTriangleDistanceType::P_B:
                return distanceSqrPointPoint(p, b);
            case PointTriangleDistanceType::P_C:
                return distanceSqrPointPoint(p, c);
            case PointTriangleDistanceType::P_AB:
                return distanceSqrPointLine(p, a, b);
            case PointTriangleDistanceType::P_BC:
                return distanceSqrPointLine(p, b, c);
            case PointTriangleDistanceType::P_CA:
                return distanceSqrPointLine(p, c, a);
            case PointTriangleDistanceType::P_ABC:
                return distanceSqrPointPlane(p, a, b, c);
            }
            return 0.0;
        }

        Real edgeEdgeDistanceSqrByType(EdgeEdgeDistanceType type,
                                       const glm::dvec3& a0,
                                       const glm::dvec3& a1,
                                       const glm::dvec3& b0,
                                       const glm::dvec3& b1)
        {
            switch (type)
            {
            case EdgeEdgeDistanceType::A_C:
                return distanceSqrPointPoint(a0, b0);
            case EdgeEdgeDistanceType::A_D:
                return distanceSqrPointPoint(a0, b1);
            case EdgeEdgeDistanceType::B_C:
                return distanceSqrPointPoint(a1, b0);
            case EdgeEdgeDistanceType::B_D:
                return distanceSqrPointPoint(a1, b1);
            case EdgeEdgeDistanceType::AB_C:
                return distanceSqrPointLine(b0, a0, a1);
            case EdgeEdgeDistanceType::AB_D:
                return distanceSqrPointLine(b1, a0, a1);
            case EdgeEdgeDistanceType::A_CD:
                return distanceSqrPointLine(a0, b0, b1);
            case EdgeEdgeDistanceType::B_CD:
                return distanceSqrPointLine(a1, b0, b1);
            case EdgeEdgeDistanceType::AB_CD:
                return distanceSqrLineLine(a0, a1, b0, b1);
            }
            return 0.0;
        }

        Real pointEdgeProjectionParameter(const glm::dvec3& p,
                                          const glm::dvec3& a,
                                          const glm::dvec3& b)
        {
            const glm::dvec3 e = b - a;
            const Real e_len2 = glm::dot(e, e);
            if (e_len2 == 0.0)
            {
                return 0.0;
            }
            return glm::dot(p - a, e) / e_len2;
        }

        std::optional<Real> runACCD(bool edgeEdge,
                                    std::array<glm::dvec3, 4> x,
                                    std::array<glm::dvec3, 4> u,
                                    Real toi,
                                    Real reserved_distance)
        {
            const glm::dvec3 mean_step = (u[0] + u[1] + u[2] + u[3]) * 0.25;
            for (glm::dvec3& step : u)
            {
                step -= mean_step;
            }

            const Real lp =
                edgeEdge
                    ? std::max(glm::length(u[0]), glm::length(u[1])) +
                    std::max(glm::length(u[2]), glm::length(u[3]))
                    : glm::length(u[0]) +
                    std::max({
                        glm::length(u[1]),
                        glm::length(u[2]),
                        glm::length(u[3])
                    });
            if (lp == 0.0)
            {
                return std::nullopt;
            }

            auto distance = [&]() -> Real
            {
                if (edgeEdge)
                {
                    const EdgeEdgeDistanceType type =
                        decideEdgeEdgeDistanceType(x[0], x[1], x[2], x[3]);
                    return std::sqrt(
                        edgeEdgeDistanceSqrByType(type, x[0], x[1], x[2], x[3]));
                }
                const PointTriangleDistanceType type =
                    decidePointTriangleDistanceType(x[0], x[1], x[2], x[3]);
                return std::sqrt(
                    pointTriangleDistanceSqrByType(type, x[0], x[1], x[2], x[3]));
            };

            Real dis = distance();
            if (dis <= reserved_distance)
            {
                return 0.0;
            }
            const Real initial_gap = dis - reserved_distance;
            const Real gap = kACCDGapFraction * initial_gap;
            Real t = 0.0;
            Real dt = (1.0 - kACCDGapFraction) * (initial_gap / lp);

            for (int iteration = 0; iteration < kMaxACCDIterations; ++iteration)
            {
                for (int i = 0; i < 4; ++i)
                {
                    x[static_cast<size_t>(i)] += u[static_cast<size_t>(i)] * dt;
                }
                dis = distance();
                if (dis - reserved_distance < gap + 1.0e-10)
                {
                    return t == 0.0 ? dt : t;
                }
                t += dt;
                if (t > toi)
                {
                    return std::nullopt;
                }
                dt = 0.9 * (dis - reserved_distance) / lp;
            }
            return t > 0.0 ? std::optional<Real>(t) : std::nullopt;
        }

        bool shouldKeepByDistanceOrCCD(Real distance_sqr,
                                       std::optional<Real> ccd_toi,
                                       const ContactDetectionConfig& config,
                                       Real primitive_thickness)
        {
            const Real threshold =
                std::max(config.detectionDistance, config.dHat) + primitive_thickness;
            return distance_sqr <= threshold * threshold ||
                (ccd_toi && *ccd_toi <= config.toi);
        }

        ContactStencil createPointPointStencil(PointIdx first,
                                               PointIdx second,
                                               const GlobalGeometryManager& geometry,
                                               const ContactDetectionConfig& config)
        {
            return makeStencil(
                ContactCase::PP,
                {first, second, -1, -1},
                config,
                pointRadius(geometry, first) + pointRadius(geometry, second));
        }

        ContactStencil createPointEdgeStencil(PointIdx point,
                                              PointIdx edge0,
                                              PointIdx edge1,
                                              const GlobalGeometryManager& geometry,
                                              const ContactDetectionConfig& config,
                                              Real fallback_edge_offset = 0.0)
        {
            return makeStencil(
                ContactCase::PE,
                {point, edge0, edge1, -1},
                config,
                pointEdgeSurfaceOffset(
                    geometry, point, edge0, edge1, fallback_edge_offset));
        }

        std::optional<ContactStencil> createPointTriangleStencil(
            PointIdx point,
            const std::array<PointIdx, 3>& triangle,
            const GlobalGeometryManager& geometry,
            std::span<const glm::dvec3> direction,
            const ContactDetectionConfig& config,
            Real primitive_thickness)
        {
            const glm::dvec3 p = geometry.worldPosition(point);
            const glm::dvec3 a = geometry.worldPosition(triangle[0]);
            const glm::dvec3 b = geometry.worldPosition(triangle[1]);
            const glm::dvec3 c = geometry.worldPosition(triangle[2]);
            const Real point_offset = pointRadius(geometry, point);
            const Real triangle_offset =
                std::max(primitive_thickness,
                         triangleSurfaceOffset(geometry, triangle));
            const PointTriangleDistanceType type =
                decidePointTriangleDistanceType(p, a, b, c);
            const Real distance_sqr = pointTriangleDistanceSqrByType(type, p, a, b, c);
            const std::optional<Real> toi =
                runACCD(false,
                        {p, a, b, c},
                        {
                            direction[point],
                            direction[triangle[0]],
                            direction[triangle[1]],
                            direction[triangle[2]]
                        },
                        config.toi,
                        point_offset + triangle_offset);
            if (!shouldKeepByDistanceOrCCD(
                distance_sqr, toi, config, point_offset + triangle_offset))
            {
                return std::nullopt;
            }

            switch (type)
            {
            case PointTriangleDistanceType::P_A:
                return createPointPointStencil(point, triangle[0], geometry, config);
            case PointTriangleDistanceType::P_B:
                return createPointPointStencil(point, triangle[1], geometry, config);
            case PointTriangleDistanceType::P_C:
                return createPointPointStencil(point, triangle[2], geometry, config);
            case PointTriangleDistanceType::P_AB:
                return createPointEdgeStencil(
                    point, triangle[0], triangle[1], geometry, config, triangle_offset);
            case PointTriangleDistanceType::P_BC:
                return createPointEdgeStencil(
                    point, triangle[1], triangle[2], geometry, config, triangle_offset);
            case PointTriangleDistanceType::P_CA:
                return createPointEdgeStencil(
                    point, triangle[2], triangle[0], geometry, config, triangle_offset);
            case PointTriangleDistanceType::P_ABC:
                return makeStencil(
                    ContactCase::PT,
                    {point, triangle[0], triangle[1], triangle[2]},
                    config,
                    point_offset + triangle_offset);
            }
            return std::nullopt;
        }

        ContactStencil createEdgeEdgeFeatureStencil(
            EdgeEdgeDistanceType type,
            const std::array<PointIdx, 2>& first,
            const std::array<PointIdx, 2>& second,
            const GlobalGeometryManager& geometry,
            const ContactDetectionConfig& config,
            Real first_offset,
            Real second_offset)
        {
            switch (type)
            {
            case EdgeEdgeDistanceType::A_C:
                return createPointPointStencil(first[0], second[0], geometry, config);
            case EdgeEdgeDistanceType::A_D:
                return createPointPointStencil(first[0], second[1], geometry, config);
            case EdgeEdgeDistanceType::B_C:
                return createPointPointStencil(first[1], second[0], geometry, config);
            case EdgeEdgeDistanceType::B_D:
                return createPointPointStencil(first[1], second[1], geometry, config);
            case EdgeEdgeDistanceType::AB_C:
                return createPointEdgeStencil(
                    second[0], first[0], first[1], geometry, config, first_offset);
            case EdgeEdgeDistanceType::AB_D:
                return createPointEdgeStencil(
                    second[1], first[0], first[1], geometry, config, first_offset);
            case EdgeEdgeDistanceType::A_CD:
                return createPointEdgeStencil(
                    first[0], second[0], second[1], geometry, config, second_offset);
            case EdgeEdgeDistanceType::B_CD:
                return createPointEdgeStencil(
                    first[1], second[0], second[1], geometry, config, second_offset);
            case EdgeEdgeDistanceType::AB_CD:
                return makeStencil(
                    ContactCase::EE,
                    {first[0], first[1], second[0], second[1]},
                    config,
                    first_offset + second_offset);
            }
            return makeStencil(ContactCase::EE, {-1, -1, -1, -1}, config);
        }

        std::optional<ContactStencil> createEdgeEdgeStencil(
            const std::array<PointIdx, 2>& first,
            const std::array<PointIdx, 2>& second,
            const GlobalGeometryManager& geometry,
            std::span<const glm::dvec3> direction,
            const ContactDetectionConfig& config,
            Real primitive_thickness)
        {
            const glm::dvec3 a0 = geometry.worldPosition(first[0]);
            const glm::dvec3 a1 = geometry.worldPosition(first[1]);
            const glm::dvec3 b0 = geometry.worldPosition(second[0]);
            const glm::dvec3 b1 = geometry.worldPosition(second[1]);
            const Real first_offset =
                edgeSurfaceOffset(geometry, first[0], first[1], primitive_thickness);
            const Real second_offset =
                edgeSurfaceOffset(geometry, second[0], second[1], primitive_thickness);
            const Real reserved_dist = first_offset + second_offset;
            const EdgeEdgeDistanceType type = decideEdgeEdgeDistanceType(a0, a1, b0, b1);
            const Real distance_sqr =
                edgeEdgeDistanceSqrByType(type, a0, a1, b0, b1);
            const std::optional<Real> toi =
                runACCD(true,
                        {a0, a1, b0, b1},
                        {
                            direction[first[0]],
                            direction[first[1]],
                            direction[second[0]],
                            direction[second[1]]
                        },
                        config.toi,
                        reserved_dist);
            if (!shouldKeepByDistanceOrCCD(
                distance_sqr, toi, config, reserved_dist))
            {
                return std::nullopt;
            }

            return createEdgeEdgeFeatureStencil(
                type, first, second, geometry, config, first_offset, second_offset);
        }

        void validateCPUCCDInput(const GlobalGeometryManager& geometry,
                                 const GeometryBuffer& direction)
        {
            if (!direction.isCPU())
            {
                throw std::runtime_error(
                    "CPU contact detection requires a CPU geometry direction buffer");
            }
            if (direction.cpu().size() < static_cast<size_t>(geometry.pointCount()))
            {
                throw std::invalid_argument(
                    "geometry direction buffer is smaller than geometry");
            }
        }

        [[nodiscard]] std::vector<PrimitiveRef> buildPointPrimitives(
            const GlobalGeometryManager& geometry,
            const GeometryBuffer& direction,
            Real toi)
        {
            std::vector<PrimitiveRef> primitives;
            primitives.reserve(static_cast<size_t>(geometry.pointCount()));
            for (int point = 0; point < geometry.pointCount(); ++point)
            {
                const Real radius = geometry.points[static_cast<size_t>(point)].radius;
                primitives.push_back(PrimitiveRef{
                    .id = point,
                    .reserved_dist = radius,
                    .bounds =
                    geometry.trajectoryPointBounds(point, direction.cpu(), toi)
                            .dilate(radius),
                });
            }
            return primitives;
        }

        [[nodiscard]] std::vector<PrimitiveRef> buildEdgePrimitives(
            const GlobalGeometryManager& geometry,
            const GeometryBuffer& direction,
            Real toi)
        {
            std::vector<PrimitiveRef> primitives;
            primitives.reserve(geometry.edgeCount());
            for (int edge = 0; edge < geometry.edgeCount(); ++edge)
            {
                const Real radius = geometry.edges[edge].radius;
                primitives.push_back(PrimitiveRef{
                    .id = edge,
                    .reserved_dist = radius,
                    .bounds =
                    geometry.trajectoryEdgeBounds(edge, direction.cpu(), toi, radius)
                            .dilate(radius),
                });
            }
            return primitives;
        }

        [[nodiscard]] std::vector<PrimitiveRef> buildTrianglePrimitives(
            const GlobalGeometryManager& geometry,
            const GeometryBuffer& direction,
            Real toi)
        {
            std::vector<PrimitiveRef> primitives;
            primitives.reserve(static_cast<size_t>(geometry.triangleCount()));
            for (int triangle = 0; triangle < geometry.triangleCount(); ++triangle)
            {
                const Real thickness = geometry.triangles[triangle].thickness;
                primitives.push_back(PrimitiveRef{
                    .id = triangle,
                    .reserved_dist = thickness,
                    .bounds =
                    geometry.trajectoryTriangleBounds(
                                triangle, direction.cpu(), toi, thickness)
                            .dilate(thickness),
                });
            }
            return primitives;
        }

        int stencilPointCount(ContactCase type)
        {
            switch (type)
            {
            case ContactCase::PP:
                return 2;
            case ContactCase::PE:
                return 3;
            case ContactCase::PT:
            case ContactCase::EE:
                return 4;
            }
            return 0;
        }

        void routeStencil(GlobalContactRouter& routed,
                          const GlobalGeometryManager& geometry_manager,
                          const ContactStencil& stencil)
        {
            const int point_count = stencilPointCount(stencil.type);
            const GeometryStencilInfo info = geometry_manager.classify(
                std::span(stencil.geometryIds.data(), point_count));
            if (info.subsystemCount == 1)
            {
                auto& subsystem_contacts = routed.subsystemInternalContacts[info.subsystems[0]];
                subsystem_contacts.subsystem = info.subsystems[0];
                subsystem_contacts.contacts.push_back(stencil);
                return;
            }
            routed.globalContacts.push_back(stencil);
        }

        std::optional<ContactStencil> lowerCandidateToBarrierStencil(
            const ContactCandidate& candidate,
            const GlobalGeometryManager& geometry,
            std::span<const glm::dvec3> direction,
            const ContactDetectionConfig& config)
        {
            switch (candidate.kind)
            {
            case EContactCandidate::PointTriangle:
                return createPointTriangleStencil(
                    PointIdx{candidate.geometryIds[0]},
                    {
                        PointIdx{candidate.geometryIds[1]},
                        PointIdx{candidate.geometryIds[2]},
                        PointIdx{candidate.geometryIds[3]}
                    },
                    geometry,
                    direction,
                    config,
                    0.0);
            case EContactCandidate::EdgeEdge:
                return createEdgeEdgeStencil(
                    {
                        PointIdx{candidate.geometryIds[0]},
                        PointIdx{candidate.geometryIds[1]}
                    },
                    {
                        PointIdx{candidate.geometryIds[2]},
                        PointIdx{candidate.geometryIds[3]}
                    },
                    geometry,
                    direction,
                    config,
                    0.0);
            case EContactCandidate::PointPoint:
                {
                    const PointIdx first{candidate.geometryIds[0]};
                    const PointIdx second{candidate.geometryIds[1]};
                    const Real reserved_dist =
                        pointRadius(geometry, first) + pointRadius(geometry, second);
                    const Real distance_sqr = distanceSqrPointPoint(
                        geometry.worldPosition(first),
                        geometry.worldPosition(second));
                    if (distance_sqr >
                        (config.dHat + reserved_dist) * (config.dHat + reserved_dist))
                    {
                        return std::nullopt;
                    }
                    return createPointPointStencil(first, second, geometry, config);
                }
            case EContactCandidate::PointEdge:
                {
                    const PointIdx point{candidate.geometryIds[0]};
                    const PointIdx edge0{candidate.geometryIds[1]};
                    const PointIdx edge1{candidate.geometryIds[2]};
                    const Real fallback_edge_offset =
                        std::max(0.0,
                                 candidate.reservedDistance - pointRadius(geometry, point));
                    const Real reserved_dist =
                        pointEdgeSurfaceOffset(
                            geometry, point, edge0, edge1, fallback_edge_offset);
                    const Real distance_sqr = distanceSqrPointLine(
                        geometry.worldPosition(point),
                        geometry.worldPosition(edge0),
                        geometry.worldPosition(edge1));
                    if (distance_sqr >
                        (config.dHat + reserved_dist) * (config.dHat + reserved_dist))
                    {
                        return std::nullopt;
                    }
                    const Real t = pointEdgeProjectionParameter(
                        geometry.worldPosition(point),
                        geometry.worldPosition(edge0),
                        geometry.worldPosition(edge1));
                    if (t <= 0.0)
                    {
                        return createPointPointStencil(point, edge0, geometry, config);
                    }
                    if (t >= 1.0)
                    {
                        return createPointPointStencil(point, edge1, geometry, config);
                    }
                    return createPointEdgeStencil(
                        point, edge0, edge1, geometry, config, fallback_edge_offset);
                }
            }
            return std::nullopt;
        }

        std::optional<Real> computeCandidateTimeOfImpact(
            const GlobalGeometryManager& geometry,
            std::span<const glm::dvec3> direction,
            const ContactCandidate& candidate,
            Real toi)
        {
            switch (candidate.kind)
            {
            case EContactCandidate::PointTriangle:
                {
                    const PointIdx point{candidate.geometryIds[0]};
                    const std::array<PointIdx, 3> triangle{
                        PointIdx{candidate.geometryIds[1]},
                        PointIdx{candidate.geometryIds[2]},
                        PointIdx{candidate.geometryIds[3]}
                    };
                    return runACCD(false,
                                   {
                                       geometry.worldPosition(point),
                                       geometry.worldPosition(triangle[0]),
                                       geometry.worldPosition(triangle[1]),
                                       geometry.worldPosition(triangle[2])
                                   },
                                   {
                                       direction[point],
                                       direction[triangle[0]],
                                       direction[triangle[1]],
                                       direction[triangle[2]]
                                   },
                                   toi,
                                   candidate.reservedDistance);
                }
            case EContactCandidate::EdgeEdge:
                {
                    const std::array<PointIdx, 2> first{
                        PointIdx{candidate.geometryIds[0]},
                        PointIdx{candidate.geometryIds[1]}
                    };
                    const std::array<PointIdx, 2> second{
                        PointIdx{candidate.geometryIds[2]},
                        PointIdx{candidate.geometryIds[3]}
                    };
                    return runACCD(true,
                                   {
                                       geometry.worldPosition(first[0]),
                                       geometry.worldPosition(first[1]),
                                       geometry.worldPosition(second[0]),
                                       geometry.worldPosition(second[1])
                                   },
                                   {
                                       direction[first[0]],
                                       direction[first[1]],
                                       direction[second[0]],
                                       direction[second[1]]
                                   },
                                   toi,
                                   candidate.reservedDistance);
                }
            case EContactCandidate::PointPoint:
            case EContactCandidate::PointEdge:
                return std::nullopt;
            }
            return std::nullopt;
        }
    } // namespace

    namespace detail
    {
        namespace
        {
            void addCandidate(ContactCandidateDetectionResult& result,
                              const GlobalGeometryManager& geometry,
                              const GeometryBuffer& geometryDirection,
                              const ContactDetectionConfig& config,
                              bool compute_step_size,
                              ContactCandidate candidate)
            {
                if (compute_step_size)
                {
                    const std::optional<Real> toi = computeCandidateTimeOfImpact(
                        geometry, geometryDirection.cpu(), candidate, config.toi);
                    if (toi)
                    {
                        result.stepSizeUpperBound =
                            std::min(result.stepSizeUpperBound, std::clamp(*toi, 0.0, 1.0));
                    }
                }
                result.candidates.push_back(std::move(candidate));
            }

            ContactCandidateDetectionResult buildCandidatesFromCollisionWorkListOnCPU(
                const GlobalGeometryManager& geometry,
                const GeometryBuffer& geometryDirection,
                const ContactDetectionConfig& config,
                const ContactWorkList& work,
                bool compute_step_size)
            {
                ContactCandidateDetectionResult result;
                validateCPUCCDInput(geometry, geometryDirection);

                for (const auto& pair : work.deformablePointTriangles)
                {
                    const PointIdx point = PointIdx{pair[0]};
                    const int triangle_id = pair[1];
                    const auto triangle = geometry.globalTriangle(triangle_id);
                    const Real primitive_thickness =
                        pointRadius(geometry, point) +
                        geometry.triangles.at(static_cast<size_t>(triangle_id)).thickness;
                    addCandidate(result,
                                 geometry,
                                 geometryDirection,
                                 config,
                                 compute_step_size,
                                 makeCandidate(
                                     EContactCandidate::PointTriangle,
                                     {point, triangle[0], triangle[1], triangle[2]},
                                     config,
                                     primitive_thickness));
                }

                for (const auto& pair : work.deformableEdgeEdges)
                {
                    const int first_id = pair[0];
                    const int second_id = pair[1];
                    const auto first = geometry.globalEdge(first_id);
                    const auto second = geometry.globalEdge(second_id);
                    const Real primitive_thickness =
                        geometry.edges.at(static_cast<size_t>(first_id)).radius +
                        geometry.edges.at(static_cast<size_t>(second_id)).radius;
                    addCandidate(result,
                                 geometry,
                                 geometryDirection,
                                 config,
                                 compute_step_size,
                                 makeCandidate(
                                     EContactCandidate::EdgeEdge,
                                     {first[0], first[1], second[0], second[1]},
                                     config,
                                     primitive_thickness));
                }

                return result;
            }

        } // namespace

        ContactWorkList gatherCollisionWorkListAlongDirectionOnCPU(
            const GlobalGeometryManager& geometry,
            const GeometryBuffer& geometryDirection,
            const ContactDetectionConfig& config)
        {
            if (config.storage == ContactDetectionStorage::Device)
            {
                throw std::runtime_error(
                    "CPU contact detection cannot produce device collision work lists");
            }
            ContactWorkList work;
            validateCPUCCDInput(geometry, geometryDirection);

            const Real query_dilation =
                std::max(config.detectionDistance, config.dHat);

            std::vector<PrimitiveRef> points =
                buildPointPrimitives(geometry, geometryDirection, config.toi);
            std::vector<PrimitiveRef> triangles =
                buildTrianglePrimitives(geometry, geometryDirection, config.toi);
            const spatify::LBVH<Real> triangle_tree = buildLBVH(triangles);

            if (!triangles.empty())
            {
                for (const PrimitiveRef& point : points)
                {
                    const spatify::BBox<Real, 3> query_bounds =
                        ensureFiniteBVHBounds(point.bounds.dilate(query_dilation));
                    triangle_tree.runSpatialQuery(
                        [&](int primitive_index)
                        {
                            const PrimitiveRef& triangle =
                                triangles[static_cast<size_t>(primitive_index)];
                            if (geometry.triangleContainsPoint(
                                triangle.id, PointIdx{point.id}))
                            {
                                return false;
                            }
                            work.deformablePointTriangles.push_back(
                                std::array<int, 2>{point.id, triangle.id});
                            return true;
                        },
                        [&](const spatify::BBox<Real, 3>& bounds)
                        {
                            return query_bounds.overlap(bounds);
                        });
                }
            }

            std::vector<PrimitiveRef> edges =
                buildEdgePrimitives(geometry, geometryDirection, config.toi);
            if (edges.empty())
            {
                finalizeCollisionWorkList(work);
                return work;
            }
            const spatify::LBVH<Real> edge_tree = buildLBVH(edges);

            for (int query_index = 0; query_index < static_cast<int>(edges.size());
                 query_index++)
            {
                const PrimitiveRef& query_edge = edges[query_index];
                const spatify::BBox<Real, 3> query_bounds =
                    ensureFiniteBVHBounds(query_edge.bounds.dilate(query_dilation));

                edge_tree.runSpatialQuery(
                    [&](int primitive_index)
                    {
                        if (primitive_index <= query_index)
                        {
                            return false;
                        }
                        const PrimitiveRef& candidate =
                            edges[static_cast<size_t>(primitive_index)];
                        if (geometry.edgesAdjacent(query_edge.id, candidate.id))
                        {
                            return false;
                        }
                        work.deformableEdgeEdges.push_back(
                            std::array<int, 2>{query_edge.id, candidate.id});
                        return true;
                    },
                    [&](const spatify::BBox<Real, 3>& bounds)
                    {
                        return query_bounds.overlap(bounds);
                    });
            }

            finalizeCollisionWorkList(work);
            return work;
        }

        ContactCandidates detectContactCandidatesAlongDirectionOnCPU(
            const GlobalGeometryManager& geometry,
            const GeometryBuffer& geometryDirection,
            const ContactDetectionConfig& config)
        {
            if (config.storage == ContactDetectionStorage::Device)
            {
                throw std::runtime_error(
                    "CPU contact detection cannot produce device contact tables");
            }
            const ContactWorkList work =
                gatherCollisionWorkListAlongDirectionOnCPU(
                    geometry, geometryDirection, config);
            return buildCandidatesFromCollisionWorkListOnCPU(
                    geometry, geometryDirection, config, work, false)
                .candidates;
        }

        ContactCandidateDetectionResult
        detectContactCandidatesAndStepSizeAlongDirectionOnCPU(
            const GlobalGeometryManager& geometry,
            const GeometryBuffer& geometryDirection,
            const ContactDetectionConfig& config)
        {
            if (config.storage == ContactDetectionStorage::Device)
            {
                throw std::runtime_error(
                    "CPU contact detection cannot produce device contact tables");
            }
            const ContactWorkList work =
                gatherCollisionWorkListAlongDirectionOnCPU(
                    geometry, geometryDirection, config);
            return buildCandidatesFromCollisionWorkListOnCPU(
                geometry, geometryDirection, config, work, true);
        }

        GlobalContactRouter refreshActiveContactsFromCandidatesOnCPU(
            const GlobalGeometryManager& geometry,
            const ContactCandidates& candidates,
            const ContactDetectionConfig& config)
        {
            GlobalContactRouter routed;
            const std::vector<glm::dvec3> zero_direction(
                static_cast<size_t>(geometry.pointCount()), glm::dvec3(0.0));
            std::vector<std::optional<ContactStencil>> active_contacts(candidates.size());
            tbb::parallel_for(
                size_t{0},
                candidates.size(),
                [&](size_t index)
                {
                    active_contacts[index] =
                        lowerCandidateToBarrierStencil(
                            candidates[index], geometry, zero_direction, config);
                });

            for (const std::optional<ContactStencil>& active : active_contacts)
            {
                if (active)
                {
                    routeStencil(routed, geometry, *active);
                }
            }
            return routed;
        }

        GlobalContactRouter runCCDOnCPU(
            const GlobalGeometryManager& geometry,
            const GeometryBuffer& geometryDirection,
            const ContactDetectionConfig& config)
        {
            const ContactWorkList work =
                gatherCollisionWorkListAlongDirectionOnCPU(
                    geometry, geometryDirection, config);
            GlobalContactRouter routed;
            validateCPUCCDInput(geometry, geometryDirection);

            std::vector<std::optional<ContactStencil>> point_triangle_contacts(
                work.deformablePointTriangles.size());
            tbb::parallel_for(
                size_t{0},
                work.deformablePointTriangles.size(),
                [&](size_t index)
                {
                    const auto& pair = work.deformablePointTriangles[index];
                    const PointIdx point = PointIdx{pair[0]};
                    const auto triangle = geometry.globalTriangle(pair[1]);
                    point_triangle_contacts[index] =
                        createPointTriangleStencil(point,
                                                   triangle,
                                                   geometry,
                                                   geometryDirection.cpu(),
                                                   config,
                                                   0.0);
                });

            for (const std::optional<ContactStencil>& active : point_triangle_contacts)
            {
                if (active)
                {
                    routeStencil(routed, geometry, *active);
                }
            }

            std::vector<std::optional<ContactStencil>> edge_edge_contacts(
                work.deformableEdgeEdges.size());
            tbb::parallel_for(
                size_t{0},
                work.deformableEdgeEdges.size(),
                [&](size_t index)
                {
                    const auto& pair = work.deformableEdgeEdges[index];
                    const auto first = geometry.globalEdge(pair[0]);
                    const auto second = geometry.globalEdge(pair[1]);
                    edge_edge_contacts[index] =
                        createEdgeEdgeStencil(first,
                                              second,
                                              geometry,
                                              geometryDirection.cpu(),
                                              config,
                                              0.0);
                });

            for (const std::optional<ContactStencil>& active : edge_edge_contacts)
            {
                if (active)
                {
                    routeStencil(routed, geometry, *active);
                }
            }

            return routed;
        }
    } // namespace detail
} // namespace ksk::runtime
