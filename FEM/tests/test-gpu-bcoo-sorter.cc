// ============================================================================
// test-gpu-bcoo-sorter.cc — GpuBcooSorter vs CPU BlockSparseMatrix<3>::sortByRow.
//
// Generates an unsorted BCOO with duplicate (row,col) pairs and integer-valued
// block entries (so duplicate sums are exact regardless of accumulation order),
// then checks:
//   * sorted rows match the CPU row-sorted order,
//   * compact segStart + sentinel match CPU m_rowSegments,
//   * the per-(row,col) accumulated 3x3 block matches the input (value-correct
//     under permutation, independent of within-row column order).
// ============================================================================
#ifdef FEM_GPU_ENABLED

#include <fem/gpu/gpu-bcoo-sorter.h>
#include <Maths/block-sparse-matrix.h>

#include <gtest/gtest.h>
#include <glm/gtc/type_ptr.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <cstring>
#include <map>
#include <random>
#include <vector>

using namespace ksk::rhi;
using namespace ksk::fem::gpu;

namespace {

struct BcooSorterFixture : public ::testing::Test {
    std::unique_ptr<Device>         device;
    std::unique_ptr<ShaderCompiler> compiler;
    std::unique_ptr<GPUBCOOSorter>  sorter;

    void SetUp() override {
        device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
        if (!device) GTEST_SKIP() << "No Vulkan device";
        compiler = device->createShaderCompiler();
        if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";
        sorter = std::make_unique<GPUBCOOSorter>(*device, *compiler);
        if (!sorter->valid()) GTEST_SKIP() << "BCOO sorter pipelines failed to compile";
    }

    BufferRef makeBuf(size_t bytes) {
        return device->createBuffer({
            .sizeBytes  = bytes,
            .visibility = BufferDesc::Visibility::DeviceLocal,
            .usage      = BufferDesc::Storage | BufferDesc::TransferSrc | BufferDesc::TransferDst,
        });
    }
    void upload(const BufferRef& dst, const void* data, size_t bytes) {
        auto st = device->createBuffer({
            .sizeBytes = bytes, .visibility = BufferDesc::Visibility::HostVisible,
            .usage = BufferDesc::TransferSrc});
        std::memcpy(st->map(), data, bytes); st->unmap();
        auto cmd = device->beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> rg{{{0, 0, bytes}}};
        cmd->copyBuffer(st, dst, rg);
        device->submitAndWait(*cmd, QueueType::Transfer);
    }
    template <class T>
    std::vector<T> download(const BufferRef& src, size_t count) {
        size_t bytes = count * sizeof(T);
        auto rb = device->createBuffer({
            .sizeBytes = bytes, .visibility = BufferDesc::Visibility::Readback,
            .usage = BufferDesc::TransferDst});
        auto cmd = device->beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> rg{{{0, 0, bytes}}};
        cmd->copyBuffer(src, rb, rg);
        device->submitAndWait(*cmd, QueueType::Transfer);
        std::vector<T> out(count);
        std::memcpy(out.data(), rb->map(), bytes);
        rb->unmap();
        return out;
    }

    void runCase(uint32_t nVerts, uint32_t nnz, unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<uint32_t> vert(0, nVerts - 1);
        std::uniform_int_distribution<int>       val(-9, 9);

        std::vector<uint32_t> row(nnz), col(nnz);
        std::vector<double>   blocks(size_t(nnz) * 9);
        for (uint32_t k = 0; k < nnz; ++k) {
            row[k] = vert(rng);
            col[k] = vert(rng);
            for (int i = 0; i < 9; ++i) blocks[k * 9 + i] = static_cast<double>(val(rng));
        }

        // ---- CPU reference: sortByRow for row order + segment offsets ----
        ksk::maths::BlockSparseMatrix<3> cpu(nVerts, nVerts);
        for (uint32_t k = 0; k < nnz; ++k) {
            glm::dmat3 b;
            std::memcpy(glm::value_ptr(b), &blocks[k * 9], 9 * sizeof(double));
            cpu.addBlock(int(row[k]), int(col[k]), b);
        }
        cpu.sortByRow();
        std::vector<uint32_t> cpuRow(nnz);
        for (uint32_t k = 0; k < nnz; ++k) cpuRow[k] = uint32_t(cpu.rowIndices()[k]);
        // Re-derive compact segment starts (BlockSparseMatrix keeps them private).
        std::vector<uint32_t> cpuSeg;
        cpuSeg.push_back(0);
        for (uint32_t k = 1; k < nnz; ++k)
            if (cpuRow[k] != cpuRow[k - 1]) cpuSeg.push_back(k);
        const uint32_t cpuNumSeg = uint32_t(cpuSeg.size());

        // ---- Reference per-(row,col) accumulated block from the raw input ----
        std::map<std::pair<uint32_t, uint32_t>, std::array<double, 9>> ref;
        for (uint32_t k = 0; k < nnz; ++k) {
            auto& acc = ref[{row[k], col[k]}];
            for (int i = 0; i < 9; ++i) acc[i] += blocks[k * 9 + i];
        }

        // ---- GPU sort ----
        auto bBlocks = makeBuf(size_t(nnz) * 9 * sizeof(double));
        auto bRow    = makeBuf(size_t(nnz) * sizeof(uint32_t));
        auto bCol    = makeBuf(size_t(nnz) * sizeof(uint32_t));
        auto bSeg    = makeBuf(size_t(nnz + 1) * sizeof(uint32_t));
        upload(bBlocks, blocks.data(), blocks.size() * sizeof(double));
        upload(bRow, row.data(), row.size() * sizeof(uint32_t));
        upload(bCol, col.data(), col.size() * sizeof(uint32_t));

        uint32_t numSeg = sorter->sort(bBlocks, bRow, bCol, bSeg, nnz);

        auto gRow    = download<uint32_t>(bRow, nnz);
        auto gCol    = download<uint32_t>(bCol, nnz);
        auto gBlocks = download<double>(bBlocks, size_t(nnz) * 9);
        auto gSeg    = download<uint32_t>(bSeg, numSeg + 1);

        // ---- Structural checks ----
        EXPECT_EQ(numSeg, cpuNumSeg);
        EXPECT_EQ(gRow, cpuRow);                 // ascending, same multiset => identical
        ASSERT_EQ(gSeg.size(), cpuSeg.size() + 1);
        for (uint32_t s = 0; s < cpuNumSeg; ++s) EXPECT_EQ(gSeg[s], cpuSeg[s]) << "seg " << s;
        EXPECT_EQ(gSeg[numSeg], nnz);            // sentinel

        // rows non-decreasing
        for (uint32_t k = 1; k < nnz; ++k) EXPECT_LE(gRow[k - 1], gRow[k]);

        // ---- Value check: accumulate GPU output per (row,col), compare to ref ----
        std::map<std::pair<uint32_t, uint32_t>, std::array<double, 9>> got;
        for (uint32_t k = 0; k < nnz; ++k) {
            auto& acc = got[{gRow[k], gCol[k]}];
            for (int i = 0; i < 9; ++i) acc[i] += gBlocks[k * 9 + i];
        }
        ASSERT_EQ(got.size(), ref.size());
        for (auto& [key, acc] : ref) {
            auto it = got.find(key);
            ASSERT_NE(it, got.end()) << "missing (" << key.first << "," << key.second << ")";
            for (int i = 0; i < 9; ++i)
                EXPECT_DOUBLE_EQ(it->second[i], acc[i])
                    << "(" << key.first << "," << key.second << ")[" << i << "]";
        }

        spdlog::info("[test-gpu-bcoo-sorter] nVerts={} nnz={} numSeg={} OK",
                     nVerts, nnz, numSeg);
    }
};

}  // namespace

TEST_F(BcooSorterFixture, SmallDense)      { runCase(8,   200,   1); }
TEST_F(BcooSorterFixture, MediumDup)       { runCase(50,  5000,  2); }
TEST_F(BcooSorterFixture, ManyVertsSparse) { runCase(2000, 20000, 3); }
TEST_F(BcooSorterFixture, SingleRow)       { runCase(1,   500,   4); }

#endif // FEM_GPU_ENABLED
// When RHI is disabled this TU is empty; GTest::gtest_main provides main().
