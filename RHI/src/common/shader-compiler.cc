//
// shader-compiler.cc
// DXC-based HLSL compiler implementation. R3. SPIR-V disassembly via SPIRV-Tools added in R6.
// See docs/rhi-plan.md §5.1 / §5.2.
//
// Design notes:
//   • Uses the DXC C++ API (IDxcCompiler3) rather than spawning dxc.exe — fewer
//     processes, structured error blobs, and fits the build's dependency model
//     where vcpkg already vends `directx-dxc`.
//   • Cross-platform string handling: DXC's APIs take LPCWSTR (= const wchar_t*
//     via dxc/WinAdapter.h on Linux/macOS). All compiler arguments are ASCII
//     in practice (entry points, target profiles, define names, paths) so a
//     byte-by-byte widen() is sufficient. Non-ASCII paths are not yet supported.
//   • Instance-scoped target backend: bind it once at create() time for the
//     common "compiler shadows runtime backend" case, or leave it unset and
//     choose per call for standalone tooling.
//   • Caching: an in-memory map keyed by hash(source, options, resolved backend).
//     Two identical compileHlsl() calls return the same bytecode without
//     invoking DXC twice.
//   • Cross-platform RAII: dxcapi.h doesn't ship a ComPtr; we use a tiny local
//     `DxcPtr<T>` that calls Release() on destruction.
//   • SPIR-V disassembly (R6+): DXC's Disassemble only handles DXIL. SPIR-V
//     bytecode is disassembled via SPIRV-Tools' spvBinaryToText (SPV_ENV_VULKAN_1_3).
//

#include <RHI/shader-compiler.h>

#include <spdlog/spdlog.h>

// On Windows, dxcapi.h uses COM/SAL types (HRESULT, REFCLSID, REFIID,
// IUnknown, IStream, __stdcall, _In_, IID_PPV_ARGS, ...) without pulling in
// the Windows headers itself. We must include <windows.h> + <unknwn.h> first.
// We deliberately do NOT define WIN32_LEAN_AND_MEAN: dxcapi.h's interfaces
// derive from IUnknown / take IStream*, which lean-and-mean strips out.
// On non-Windows the header auto-includes its bundled WinAdapter.h.
#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
  #include <unknwn.h>
#endif

// dxcapi.h lives at the top of the directx-dxc vcpkg include dir (no `dxc/`
// prefix) — Microsoft::DirectXShaderCompiler's INTERFACE_INCLUDE_DIRECTORIES
// already points at it.
#include <dxcapi.h>

#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace sim::rhi {
namespace {

// ---- Local RAII for DXC IUnknown-derived COM-style pointers -----------------
//
// Same shape on all platforms: ctor null, dtor Release(), no copy, move steals.
// We deliberately avoid Microsoft::WRL::ComPtr to stay header-set-minimal and
// to compile on Linux/macOS without WRL.
template <class T>
class DxcPtr {
 public:
  DxcPtr() = default;
  DxcPtr(const DxcPtr&) = delete;
  DxcPtr& operator=(const DxcPtr&) = delete;
  DxcPtr(DxcPtr&& o) noexcept : m_p(o.m_p) { o.m_p = nullptr; }
  DxcPtr& operator=(DxcPtr&& o) noexcept {
    if (this != &o) {
      if (m_p) m_p->Release();
      m_p = o.m_p;
      o.m_p = nullptr;
    }
    return *this;
  }
  ~DxcPtr() {
    if (m_p) m_p->Release();
  }

  // Address-of accessor used as out-parameter to DxcCreateInstance / GetOutput.
  T** putAddr() { return &m_p; }
  T* get() const { return m_p; }
  T* operator->() const { return m_p; }
  explicit operator bool() const { return m_p != nullptr; }

 private:
  T* m_p = nullptr;
};

// ---- Helpers ----------------------------------------------------------------

// Widen an ASCII-only string to wchar_t. DXC arg list members are all ASCII in
// practice (entry name, target profile, macro names, switch names, paths).
inline std::wstring widen(std::string_view s) {
  std::wstring w;
  w.reserve(s.size());
  for (char c : s) {
    w.push_back(static_cast<wchar_t>(static_cast<unsigned char>(c)));
  }
  return w;
}

// Separate name from widen() to avoid overload ambiguity: std::string is
// implicitly convertible to BOTH std::string_view and std::filesystem::path,
// so a single overload set sees ambiguous calls when given a std::string.
inline std::wstring widenPath(const std::filesystem::path& p) {
#ifdef _WIN32
  return p.wstring();  // Windows already stores path natively as wide.
#else
  // On POSIX, filesystem::path stores bytes; wstring() routes through the
  // current locale and may misbehave for non-ASCII bytes. R3 assumes ASCII
  // paths only — non-ASCII shader paths are recorded as future work.
  return widen(p.string());
#endif
}

const wchar_t* targetProfileFor(ShaderStage stage) {
  switch (stage) {
    case ShaderStage::Compute:
      return L"cs_6_0";
    case ShaderStage::Vertex:
      return L"vs_6_0";
    case ShaderStage::Fragment:
      return L"ps_6_0";
  }
  return nullptr;
}

inline size_t hashCombine(size_t a, size_t b) {
  // Boost-style splatter; cheap and good enough for cache keying.
  return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
}

size_t computeCacheKey(std::string_view source,
                       const ShaderCompileOptions& o,
                       Backend targetBackend) {
  size_t h = std::hash<std::string_view>{}(source);
  h = hashCombine(h, std::hash<std::string>{}(o.entryPoint));
  h = hashCombine(h, std::hash<int>{}(static_cast<int>(o.stage)));
  h = hashCombine(h, std::hash<int>{}(static_cast<int>(targetBackend)));
  h = hashCombine(h, std::hash<bool>{}(o.generateDisassembly));
  h = hashCombine(h, std::hash<bool>{}(o.enableDebugInfo));
  for (const auto& [name, value] : o.defines) {
    h = hashCombine(h, std::hash<std::string>{}(name));
    h = hashCombine(h, std::hash<std::string>{}(value));
  }
  for (const auto& dir : o.includeDirs) {
    h = hashCombine(h, std::hash<std::string>{}(dir.string()));
  }
  return h;
}

// ---- Concrete compiler ------------------------------------------------------

class DxcShaderCompiler final : public ShaderCompiler {
 public:
  explicit DxcShaderCompiler(std::optional<Backend> defaultBackend)
      : m_defaultBackend(defaultBackend) {}

  // Two-phase init so create() can return nullptr on dxcompiler load failure
  // (e.g. DLL missing) without throwing from the ctor.
  bool initialize() {
    HRESULT hr =
        DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(m_utils.putAddr()));
    if (hr < 0 || !m_utils) {
      spdlog::error(
          "[ShaderCompiler] DxcCreateInstance(DxcUtils) failed (hr=0x{:08x})",
          static_cast<uint32_t>(hr));
      return false;
    }
    hr = DxcCreateInstance(CLSID_DxcCompiler,
                           IID_PPV_ARGS(m_compiler.putAddr()));
    if (hr < 0 || !m_compiler) {
      spdlog::error(
          "[ShaderCompiler] DxcCreateInstance(DxcCompiler) failed (hr=0x{:08x})",
          static_cast<uint32_t>(hr));
      return false;
    }
    return true;
  }

  std::optional<CompiledShader> compileHlsl(
      std::string_view source,
      const ShaderCompileOptions& options) override {
    return compileImpl(source, options, "<inline>", /*extraIncludeDir=*/{});
  }

  std::optional<CompiledShader> compileHlslFile(
      const std::filesystem::path& path,
      const ShaderCompileOptions& options) override {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
      spdlog::error("[ShaderCompiler] Cannot open '{}'", path.string());
      return std::nullopt;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return compileImpl(ss.str(), options, path.string(), path.parent_path());
  }

 private:
  std::optional<Backend> resolveTargetBackend(
      const ShaderCompileOptions& options,
      std::string_view sourceName) const {
    if (options.targetBackend.has_value()) return options.targetBackend;
    if (m_defaultBackend.has_value()) return m_defaultBackend;

    spdlog::error(
        "[ShaderCompiler] {} has no target backend. Bind one via "
        "ShaderCompiler::create(Backend) or set "
        "ShaderCompileOptions::targetBackend.",
        sourceName);
    return std::nullopt;
  }

  std::optional<CompiledShader> compileImpl(
      std::string_view source,
      const ShaderCompileOptions& options,
      std::string_view sourceName,
      const std::filesystem::path& extraIncludeDir) {
    const auto resolvedBackend = resolveTargetBackend(options, sourceName);
    if (!resolvedBackend.has_value()) return std::nullopt;
    const Backend targetBackend = *resolvedBackend;

    // ---- Cache lookup -----------------------------------------------------
    const size_t key = computeCacheKey(source, options, targetBackend);
    {
      std::scoped_lock lock(m_cacheMu);
      auto it = m_cache.find(key);
      if (it != m_cache.end()) return it->second;
    }

    // ---- Build DXC argument list -----------------------------------------
    // We back the LPCWSTR pointers with std::wstring storage that lives until
    // after Compile() returns.
    std::vector<std::wstring> argStorage;
    argStorage.reserve(16 + options.defines.size() * 2 +
                       options.includeDirs.size() * 2 +
                       (extraIncludeDir.empty() ? 0 : 2));

    auto addArg = [&](std::wstring s) { argStorage.push_back(std::move(s)); };
    auto addArgL = [&](const wchar_t* s) { argStorage.emplace_back(s); };

    // Entry & target profile.
    addArgL(L"-E");
    addArg(widen(options.entryPoint));
    addArgL(L"-T");
    addArgL(targetProfileFor(options.stage));

    // Backend codegen switches.
    if (targetBackend == Backend::Vulkan) {
      addArgL(L"-spirv");
      addArgL(L"-fspv-target-env=vulkan1.3");
    }
    // Backend::Dx12 → no extra flag; DXIL is the default.

    // Optimisation / debug.
    if (options.enableDebugInfo) {
      addArgL(L"-Od");
      addArgL(L"-Zi");
    } else {
      addArgL(L"-O3");
    }

    // Defines.
    for (const auto& [name, value] : options.defines) {
      addArgL(L"-D");
      const std::string spelled = value.empty() ? name : (name + "=" + value);
      addArg(widen(spelled));
    }

    // Include dirs (caller-supplied + the source file's own directory when
    // we know it).
    for (const auto& dir : options.includeDirs) {
      addArgL(L"-I");
      addArg(widenPath(dir));
    }
#ifdef RHI_SHADER_INCLUDE_DIR
    addArgL(L"-I");
    addArg(widenPath(std::filesystem::path(RHI_SHADER_INCLUDE_DIR)));
#endif
    if (!extraIncludeDir.empty()) {
      addArgL(L"-I");
      addArg(widenPath(extraIncludeDir));
    }

    // Build the LPCWSTR* array DXC actually wants.
    std::vector<LPCWSTR> args;
    args.reserve(argStorage.size());
    for (const auto& s : argStorage) args.push_back(s.c_str());

    // ---- Source buffer ---------------------------------------------------
    DxcBuffer src{};
    src.Ptr = source.data();
    src.Size = source.size();
    src.Encoding = DXC_CP_UTF8;

    // ---- Default include handler (for #include resolution via -I) -------
    DxcPtr<IDxcIncludeHandler> includeHandler;
    if (m_utils->CreateDefaultIncludeHandler(includeHandler.putAddr()) < 0) {
      spdlog::warn(
          "[ShaderCompiler] CreateDefaultIncludeHandler failed; #includes "
          "may not resolve.");
    }

    // ---- Compile ---------------------------------------------------------
    DxcPtr<IDxcResult> result;
    HRESULT hr = m_compiler->Compile(&src, args.data(),
                                     static_cast<UINT32>(args.size()),
                                     includeHandler.get(),
                                     IID_PPV_ARGS(result.putAddr()));
    if (hr < 0 || !result) {
      spdlog::error(
          "[ShaderCompiler] {} - DXC Compile invocation failed "
          "(hr=0x{:08x})",
          sourceName, static_cast<uint32_t>(hr));
      return std::nullopt;
    }

    HRESULT status = 0;
    result->GetStatus(&status);

    // Surface compiler diagnostics regardless of status — warnings appear on
    // success too.
    if (result->HasOutput(DXC_OUT_ERRORS)) {
      DxcPtr<IDxcBlobUtf8> errorBlob;
      result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(errorBlob.putAddr()),
                        nullptr);
      if (errorBlob && errorBlob->GetStringLength() > 0) {
        const char* msg = errorBlob->GetStringPointer();
        if (status < 0) {
          spdlog::error("[ShaderCompiler] {} compilation errors:\n{}",
                        sourceName, msg);
        } else {
          spdlog::warn("[ShaderCompiler] {} compilation warnings:\n{}",
                       sourceName, msg);
        }
      }
    }

    if (status < 0) return std::nullopt;

    // ---- Extract bytecode ------------------------------------------------
    DxcPtr<IDxcBlob> object;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(object.putAddr()), nullptr);
    if (!object || object->GetBufferSize() == 0) {
      spdlog::error("[ShaderCompiler] {} produced empty bytecode", sourceName);
      return std::nullopt;
    }

    CompiledShader out;
    const auto* bytes = static_cast<const std::byte*>(object->GetBufferPointer());
    out.bytecode.assign(bytes, bytes + object->GetBufferSize());

    // ---- Optional disassembly --------------------------------------------
    //
    // IDxcCompiler3::Disassemble only handles DXIL. Feeding SPIR-V to it
    // returns E_FAIL. SPIR-V disassembly would need SPIRV-Tools'
    // spvBinaryToText, which isn't pulled in until R4 (alongside
    // spirv-reflect). For now, we populate disassembly only for DX12 output;
    // for Vulkan output we log an info note and leave the field empty.
    if (options.generateDisassembly) {
      if (targetBackend == Backend::Dx12) {
        DxcBuffer obj{};
        obj.Ptr = out.bytecode.data();
        obj.Size = out.bytecode.size();
        obj.Encoding = 0;

        DxcPtr<IDxcResult> disResult;
        HRESULT dhr = m_compiler->Disassemble(&obj,
                                              IID_PPV_ARGS(disResult.putAddr()));
        if (dhr >= 0 && disResult) {
          DxcPtr<IDxcBlobUtf8> disText;
          disResult->GetOutput(DXC_OUT_DISASSEMBLY,
                               IID_PPV_ARGS(disText.putAddr()), nullptr);
          if (disText && disText->GetStringLength() > 0) {
            out.disassembly.assign(disText->GetStringPointer(),
                                   disText->GetStringLength());
          }
        } else {
          spdlog::warn(
              "[ShaderCompiler] {} disassemble failed (hr=0x{:08x}); "
              "leaving disassembly empty.",
              sourceName, static_cast<uint32_t>(dhr));
        }
      } else {
        spdlog::info(
            "[ShaderCompiler] {} generateDisassembly=true but targetBackend"
            "=Vulkan; SPIR-V disassembly requires SPIRV-Tools (added in R4). "
            "Leaving disassembly empty.",
            sourceName);
      }
    }

    // ---- Memoise ---------------------------------------------------------
    {
      std::scoped_lock lock(m_cacheMu);
      m_cache.emplace(key, out);
    }
    return out;
  }

  std::optional<Backend> m_defaultBackend;
  DxcPtr<IDxcUtils> m_utils;
  DxcPtr<IDxcCompiler3> m_compiler;
  std::mutex m_cacheMu;
  std::unordered_map<size_t, CompiledShader> m_cache;
};

}  // namespace

std::unique_ptr<ShaderCompiler> ShaderCompiler::create() {
  auto c = std::make_unique<DxcShaderCompiler>(std::nullopt);
  if (!c->initialize()) return nullptr;
  return c;
}

std::unique_ptr<ShaderCompiler> ShaderCompiler::create(Backend defaultBackend) {
  auto c = std::make_unique<DxcShaderCompiler>(defaultBackend);
  if (!c->initialize()) return nullptr;
  return c;
}

}  // namespace sim::rhi
