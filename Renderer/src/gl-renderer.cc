#include "gl-renderer.h"
#include <glad/glad.h>      // Must come before GLFW
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <cmath>
#include <limits>

namespace sim::renderer {

// Shader source code
static const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec3 aColor;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    out vec3 FragPos;
    out vec3 Normal;
    out vec3 VertexColor;

    void main() {
        vec4 worldPos = model * vec4(aPos, 1.0);
        FragPos = worldPos.xyz;
        Normal = mat3(transpose(inverse(model))) * aNormal;
        VertexColor = aColor;
        gl_Position = projection * view * worldPos;
    }
)";

static const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;

    in vec3 FragPos;
    in vec3 Normal;
    in vec3 VertexColor;

    uniform vec3 lightPos;
    uniform vec3 lightPos2;
    uniform vec3 viewPos;
    uniform vec3 objectColor;

    // Blinn-Phong 单光源计算
    vec3 calcBlinnPhong(vec3 lightDir, vec3 viewDir, vec3 norm,
                        vec3 baseColor, vec3 lightColor, float intensity) {
        // Diffuse (半 Lambert wrap — 暗面不完全黑)
        float NdotL = dot(norm, lightDir);
        float diff = max(NdotL, 0.0);
        // wrap lighting: 把 [-1,1] 映射到 [0.2, 1.0]，暗面保留一点光
        float wrapDiff = diff * 0.8 + 0.2;
        vec3 diffuse = wrapDiff * baseColor * lightColor * intensity;

        // Specular — Blinn-Phong half-vector
        vec3 halfDir = normalize(lightDir + viewDir);
        float NdotH = max(dot(norm, halfDir), 0.0);
        // shininess = 32: 适中的高光大小，肉眼明显可见
        float spec = pow(NdotH, 32.0);
        // 只在正面才有高光（避免背面穿透）
        spec *= step(0.0, NdotL);
        vec3 specular = spec * lightColor * intensity * 0.5;

        return diffuse + specular;
    }

    void main() {
        // 材质基础色
        vec3 baseColor = (dot(VertexColor, VertexColor) > 0.001)
                         ? VertexColor : objectColor;

        vec3 norm = normalize(Normal);
        vec3 viewDir = normalize(viewPos - FragPos);

        // --- Ambient（环境光，模拟天光 — 上方偏暖，下方偏冷）---
        float upFactor = dot(norm, vec3(0.0, 1.0, 0.0)) * 0.5 + 0.5; // [0, 1]
        vec3 ambientColor = mix(vec3(0.08, 0.10, 0.14),   // 地面反射冷色
                                vec3(0.18, 0.16, 0.14),    // 天光暖色
                                upFactor);
        vec3 ambient = ambientColor * baseColor;

        // --- 主光源（Key light）---
        vec3 lightDir1 = normalize(lightPos - FragPos);
        vec3 key = calcBlinnPhong(lightDir1, viewDir, norm,
                                  baseColor, vec3(1.0, 0.95, 0.9), 1.0);

        // --- 填充光（Fill light）---
        vec3 lightDir2 = normalize(lightPos2 - FragPos);
        vec3 fill = calcBlinnPhong(lightDir2, viewDir, norm,
                                   baseColor, vec3(0.85, 0.9, 1.0), 0.35);

        // --- Fresnel 边缘光（让轮廓从背景中脱出）---
        float fresnel = 1.0 - max(dot(norm, viewDir), 0.0);
        fresnel = pow(fresnel, 3.0) * 0.25;
        vec3 rim = fresnel * vec3(0.6, 0.7, 0.8);

        // --- 合成 ---
        vec3 result = ambient + key + fill + rim;

        // Gamma 校正（线性空间 → sRGB）
        result = pow(result, vec3(1.0 / 2.2));

        FragColor = vec4(result, 1.0);
    }
)";

// Particle shader — simple point sprites with depth
static const char* particleVertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;

    uniform mat4 view;
    uniform mat4 projection;
    uniform float pointSize;      // world-space radius
    uniform float viewportHeight;

    void main() {
        vec4 viewPos = view * vec4(aPos, 1.0);
        gl_Position = projection * viewPos;
        // Physically-correct perspective point size:
        // projection[1][1] = 1/tan(fov/2), so screen pixels = radius * proj[1][1] * viewportHeight / dist
        float dist = -viewPos.z;  // view-space depth (positive towards camera)
        gl_PointSize = max(1.0, pointSize * projection[1][1] * viewportHeight / dist);
    }
)";

static const char* particleFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    uniform vec3 particleColor;

    void main() {
        // Discard corners to make a circle
        vec2 coord = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(coord, coord);
        if (r2 > 1.0) discard;

        // Simple fake sphere shading
        float shade = 1.0 - r2 * 0.5;
        FragColor = vec4(particleColor * shade, 1.0);
    }
)";

// Wireframe shader
static const char* wireVertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    uniform mat4 view;
    uniform mat4 projection;
    void main() {
        gl_Position = projection * view * vec4(aPos, 1.0);
    }
)";

static const char* wireFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    uniform vec3 wireColor;
    void main() {
        FragColor = vec4(wireColor, 1.0);
    }
)";

// Helper function to compile shader
static unsigned int compileShader(unsigned int type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation error:\n" << infoLog << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

// Helper function to create shader program
static unsigned int createShaderProgram(const char* vertexSrc, const char* fragmentSrc) {
    unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexSrc);
    unsigned int fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);

    unsigned int program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader program linking error:\n" << infoLog << std::endl;
        glDeleteProgram(program);
        program = 0;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
}

void GLRenderer::initialize(const RendererConfig& config) {
    m_config = config;

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    if (config.headless) {
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    }

    m_window = glfwCreateWindow(config.windowWidth, config.windowHeight,
                                config.windowTitle.c_str(), nullptr, nullptr);
    if (!m_window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(m_window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    if (config.vsync) {
        glfwSwapInterval(1);
    } else {
        glfwSwapInterval(0);
    }

    // Compile shaders
    m_meshShader = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    m_wireShader = createShaderProgram(wireVertexShaderSource, wireFragmentShaderSource);
    m_particleShader = createShaderProgram(particleVertexShaderSource, particleFragmentShaderSource);

    // Set up OpenGL state
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_PROGRAM_POINT_SIZE);
    // 浅灰白背景（专业软件风，干净中性）
    glClearColor(0.92f, 0.92f, 0.94f, 1.0f);

    // Set up input callbacks
    setupInputCallbacks();

    std::cout << "GLRenderer initialized successfully" << std::endl;
}

void GLRenderer::drawFrame(const SceneProxy& scene) {
    // === 首帧自适应初始化（之后完全由用户控制）===
    if (!m_cameraInitialized) {
        auto bounds = computeSceneBounds(scene);
        m_camera.target = bounds.center;

        // 根据包围球和 FOV 计算初始观察距离
        float halfFovTan = std::tan(glm::radians(m_camera.fov * 0.5f));
        m_distance = (bounds.radius / halfFovTan) * 1.6f;
        m_distance = std::max(m_distance, bounds.radius * 4.0f);

        m_pitch = 25.0f;
        m_yaw = -60.0f;
        m_cameraInitialized = true;

        updateCameraFromInput();
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Use mesh shader
    glUseProgram(m_meshShader);

    // 相机相关矩阵
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = computeViewMatrix();
    glm::mat4 projection = computeProjectionMatrix();

    glUniformMatrix4fv(glGetUniformLocation(m_meshShader, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(m_meshShader, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(m_meshShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    // 光照 — 光源位置基于相机距离缩放
    float lightScale = std::max(m_distance * 0.6f, 3.0f);
    glm::vec3 lightPos = m_camera.target + glm::vec3(lightScale * 0.6f, lightScale, lightScale * 0.8f);
    glUniform3fv(glGetUniformLocation(m_meshShader, "lightPos"), 1, glm::value_ptr(lightPos));

    // 填充光（另一侧，较远，补暗面）
    glm::vec3 lightPos2 = m_camera.target + glm::vec3(-lightScale * 0.5f, lightScale * 0.3f, -lightScale * 0.7f);
    glUniform3fv(glGetUniformLocation(m_meshShader, "lightPos2"), 1, glm::value_ptr(lightPos2));

    // 视点
    glUniform3fv(glGetUniformLocation(m_meshShader, "viewPos"), 1, glm::value_ptr(m_camera.position));

    // 默认材质色（温暖的陶土/粘土色）
    glm::vec3 defaultColor(0.75f, 0.55f, 0.38f);
    glUniform3fv(glGetUniformLocation(m_meshShader, "objectColor"), 1, glm::value_ptr(defaultColor));

    // Draw meshes
    for (const auto& mesh : scene.meshes) {
        drawMesh(mesh);
    }

    // Draw wireframes
    if (!scene.wireframes.empty()) {
        glUseProgram(m_wireShader);
        glUniformMatrix4fv(glGetUniformLocation(m_wireShader, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(m_wireShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        for (const auto& wire : scene.wireframes) {
            drawWireframe(wire);
        }
    }

    // Draw particles
    for (const auto& particle : scene.particles) {
        drawParticles(particle);
    }

    // 绘制地面参考网格
    drawGroundGrid(view, projection);
}

void GLRenderer::uploadMesh(const MeshProxy& mesh) {
    auto it = m_meshCache.find(mesh.name);
    bool needCreate = (it == m_meshCache.end());

    GLMeshState state;
    if (!needCreate) {
        state = it->second;
        // Check if we need to reallocate
        if (state.vertexCount != mesh.positions.size()) {
            // Delete old buffers
            glDeleteVertexArrays(1, &state.vao);
            glDeleteBuffers(1, &state.vbo);
            glDeleteBuffers(1, &state.ebo);
            if (state.nbo) glDeleteBuffers(1, &state.nbo);
            needCreate = true;
        }
    }

    if (needCreate) {
        // Create new buffers
        glGenVertexArrays(1, &state.vao);
        glGenBuffers(1, &state.vbo);
        glGenBuffers(1, &state.ebo);
        glGenBuffers(1, &state.nbo);
        state.vertexCount = mesh.positions.size();
    }

    state.indexCount = static_cast<int>(mesh.triangles.size() * 3);

    // Upload data
    glBindVertexArray(state.vao);

    // Positions
    glBindBuffer(GL_ARRAY_BUFFER, state.vbo);
    glBufferData(GL_ARRAY_BUFFER, mesh.positions.size() * sizeof(core::Vec3f),
                 mesh.positions.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(core::Vec3f), (void*)0);
    glEnableVertexAttribArray(0);

    // Normals
    glBindBuffer(GL_ARRAY_BUFFER, state.nbo);
    if (!mesh.normals.empty()) {
        glBufferData(GL_ARRAY_BUFFER, mesh.normals.size() * sizeof(core::Vec3f),
                     mesh.normals.data(), GL_DYNAMIC_DRAW);
    } else {
        // 安全后备：全部朝上的法线（保证不全黑）
        std::vector<core::Vec3f> defaultNormals(mesh.positions.size(), core::Vec3f(0.0f, 1.0f, 0.0f));
        glBufferData(GL_ARRAY_BUFFER, defaultNormals.size() * sizeof(core::Vec3f),
                     defaultNormals.data(), GL_DYNAMIC_DRAW);
    }
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(core::Vec3f), (void*)0);
    glEnableVertexAttribArray(1);

    // Indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, state.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.triangles.size() * sizeof(core::Vec3u),
                 mesh.triangles.data(), GL_DYNAMIC_DRAW);

    glBindVertexArray(0);

    m_meshCache[mesh.name] = state;
}

void GLRenderer::drawMesh(const MeshProxy& mesh) {
    uploadMesh(mesh);

    auto it = m_meshCache.find(mesh.name);
    if (it == m_meshCache.end()) return;

    const auto& state = it->second;

    glUseProgram(m_meshShader);

    // Per-object color: 负值表示使用全局默认色
    if (mesh.objectColor.x >= 0.0f) {
        glUniform3fv(glGetUniformLocation(m_meshShader, "objectColor"), 1,
                     glm::value_ptr(mesh.objectColor));
    }

    glBindVertexArray(state.vao);
    glDrawElements(GL_TRIANGLES, state.indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void GLRenderer::drawWireframe(const WireframeProxy& wf) {
    if (wf.positions.empty() || wf.edges.empty() || !m_wireShader) return;

    auto it = m_meshCache.find(wf.name);
    bool needCreate = (it == m_meshCache.end());

    GLMeshState state{};
    if (!needCreate) {
        state = it->second;
        const auto indexCount = static_cast<int>(wf.edges.size() * 2);
        if (state.vertexCount != wf.positions.size() ||
            state.indexCount != indexCount) {
            glDeleteVertexArrays(1, &state.vao);
            glDeleteBuffers(1, &state.vbo);
            glDeleteBuffers(1, &state.ebo);
            needCreate = true;
        }
    }

    if (needCreate) {
        glGenVertexArrays(1, &state.vao);
        glGenBuffers(1, &state.vbo);
        glGenBuffers(1, &state.ebo);
        state.vertexCount = wf.positions.size();
    }
    state.indexCount = static_cast<int>(wf.edges.size() * 2);

    glBindVertexArray(state.vao);

    glBindBuffer(GL_ARRAY_BUFFER, state.vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(wf.positions.size() * sizeof(core::Vec3f)),
                 wf.positions.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(core::Vec3f), nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, state.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(wf.edges.size() * sizeof(core::Vec2u)),
                 wf.edges.data(), GL_DYNAMIC_DRAW);

    glBindVertexArray(0);
    m_meshCache[wf.name] = state;

    glUseProgram(m_wireShader);
    glUniform3fv(glGetUniformLocation(m_wireShader, "wireColor"), 1,
                 glm::value_ptr(wf.color));

    glLineWidth(2.0f);
    glBindVertexArray(state.vao);
    glDrawElements(GL_LINES, state.indexCount, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
    glLineWidth(1.0f);
}

void GLRenderer::drawParticles(const ParticleProxy& particles) {
    if (particles.positions.empty() || !m_particleShader) return;

    // Lazily create / update particle GPU buffer (keyed by name, same pattern as meshes)
    auto it = m_meshCache.find(particles.name);
    bool needCreate = (it == m_meshCache.end());

    GLMeshState state{};
    if (!needCreate) {
        state = it->second;
        if (state.vertexCount != particles.positions.size()) {
            glDeleteVertexArrays(1, &state.vao);
            glDeleteBuffers(1, &state.vbo);
            needCreate = true;
        }
    }
    if (needCreate) {
        glGenVertexArrays(1, &state.vao);
        glGenBuffers(1, &state.vbo);
        state.vertexCount = particles.positions.size();
    }

    glBindVertexArray(state.vao);
    glBindBuffer(GL_ARRAY_BUFFER, state.vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(particles.positions.size() * sizeof(core::Vec3f)),
                 particles.positions.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(core::Vec3f), nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    m_meshCache[particles.name] = state;

    // Draw
    glUseProgram(m_particleShader);

    glm::mat4 view       = computeViewMatrix();
    glm::mat4 projection = computeProjectionMatrix();
    glUniformMatrix4fv(glGetUniformLocation(m_particleShader, "view"),       1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(m_particleShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    int width, height;
    glfwGetWindowSize(m_window, &width, &height);
    glUniform1f(glGetUniformLocation(m_particleShader, "pointSize"),      particles.radius);
    glUniform1f(glGetUniformLocation(m_particleShader, "viewportHeight"), static_cast<float>(height));
    glUniform3fv(glGetUniformLocation(m_particleShader, "particleColor"), 1, &particles.color.x);

    glBindVertexArray(state.vao);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(particles.positions.size()));
    glBindVertexArray(0);
}

glm::mat4 GLRenderer::computeViewMatrix() const {
    return glm::lookAt(m_camera.position, m_camera.target, m_camera.up);
}

glm::mat4 GLRenderer::computeProjectionMatrix() const {
    int width, height;
    glfwGetWindowSize(m_window, &width, &height);
    float aspect = (float)width / (float)height;
    return glm::perspective(glm::radians(m_camera.fov), aspect, m_camera.nearPlane, m_camera.farPlane);
}

bool GLRenderer::pollAndSwap() {
    if (!m_window) return false;

    glfwPollEvents();

    if (glfwWindowShouldClose(m_window)) {
        return false;
    }

    updateCameraFromInput();
    glfwSwapBuffers(m_window);
    return true;
}

void GLRenderer::cleanup() {
    for (auto& [name, state] : m_meshCache) {
        glDeleteVertexArrays(1, &state.vao);
        glDeleteBuffers(1, &state.vbo);
        glDeleteBuffers(1, &state.ebo);
        if (state.nbo) glDeleteBuffers(1, &state.nbo);
    }
    m_meshCache.clear();

    if (m_meshShader) glDeleteProgram(m_meshShader);
    if (m_wireShader) glDeleteProgram(m_wireShader);
    if (m_particleShader) glDeleteProgram(m_particleShader);

    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }

    glfwTerminate();
    std::cout << "GLRenderer cleaned up" << std::endl;
}

void GLRenderer::setupInputCallbacks() {
    if (!m_window) return;

    glfwSetWindowUserPointer(m_window, this);

    // 键盘回调：R 重置相机，T 切换自动跟踪
    glfwSetKeyCallback(m_window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        auto* renderer = static_cast<GLRenderer*>(glfwGetWindowUserPointer(window));
        if (action != GLFW_PRESS) return;

        switch (key) {
            case GLFW_KEY_R:  // Reset camera
                renderer->m_cameraInitialized = false;
                break;

            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
        }
    });

    glfwSetMouseButtonCallback(m_window, [](GLFWwindow* window, int button, int action, int mods) {
        auto* renderer = static_cast<GLRenderer*>(glfwGetWindowUserPointer(window));
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            renderer->m_mousePressed = (action == GLFW_PRESS);
            if (renderer->m_mousePressed) {
                glfwGetCursorPos(window, &renderer->m_lastMouseX, &renderer->m_lastMouseY);
            }
        }
    });

    glfwSetCursorPosCallback(m_window, [](GLFWwindow* window, double xpos, double ypos) {
        auto* renderer = static_cast<GLRenderer*>(glfwGetWindowUserPointer(window));
        if (renderer->m_mousePressed) {
            double dx = xpos - renderer->m_lastMouseX;
            double dy = ypos - renderer->m_lastMouseY;

            renderer->m_yaw -= (float)dx * 0.5f;
            renderer->m_pitch -= (float)dy * 0.5f;

            // Clamp pitch
            renderer->m_pitch = std::max(-89.0f, std::min(89.0f, renderer->m_pitch));

            renderer->m_lastMouseX = xpos;
            renderer->m_lastMouseY = ypos;
        }
    });

    glfwSetScrollCallback(m_window, [](GLFWwindow* window, double xoffset, double yoffset) {
        auto* renderer = static_cast<GLRenderer*>(glfwGetWindowUserPointer(window));
        // 缩放步长与当前距离成比例（15% 每 tick），保证远近手感一致
        float zoomFactor = 1.0f - (float)yoffset * 0.15f;
        renderer->m_distance *= zoomFactor;
        renderer->m_distance = std::max(0.01f, renderer->m_distance);
    });
}

void GLRenderer::updateCameraFromInput() {
    // Update camera position based on yaw, pitch, distance
    float yawRad = glm::radians(m_yaw);
    float pitchRad = glm::radians(m_pitch);

    m_camera.position.x = m_camera.target.x + m_distance * cos(pitchRad) * cos(yawRad);
    m_camera.position.y = m_camera.target.y + m_distance * sin(pitchRad);
    m_camera.position.z = m_camera.target.z + m_distance * cos(pitchRad) * sin(yawRad);

    // Call user callback if set
    if (m_inputCallback) {
        m_inputCallback(m_camera);
    }
}

// --- SceneBounds & Camera Tracking ---

SceneBounds GLRenderer::computeSceneBounds(const SceneProxy& scene) const {
    SceneBounds bounds;

    if (scene.meshes.empty() && scene.particles.empty()) return bounds;

    glm::vec3 minP(std::numeric_limits<float>::max());
    glm::vec3 maxP(std::numeric_limits<float>::lowest());

    for (const auto& mesh : scene.meshes) {
        for (const auto& p : mesh.positions) {
            glm::vec3 pos(p.x, p.y, p.z);
            minP = glm::min(minP, pos);
            maxP = glm::max(maxP, pos);
        }
    }

    for (const auto& ps : scene.particles) {
        for (const auto& p : ps.positions) {
            glm::vec3 pos(p.x, p.y, p.z);
            minP = glm::min(minP, pos);
            maxP = glm::max(maxP, pos);
        }
    }

    if (minP.x > maxP.x) return bounds; // 无有效顶点

    bounds.center = (minP + maxP) * 0.5f;
    bounds.radius = glm::length(maxP - bounds.center);
    bounds.radius = std::max(bounds.radius, 0.1f);

    return bounds;
}

// --- Ground Grid ---

void GLRenderer::buildGroundGrid() {
    const int halfExtent = 5;       // -5 到 +5
    const float spacing = 0.5f;
    std::vector<glm::vec3> lines;

    for (int i = -halfExtent; i <= halfExtent; i++) {
        float fi = static_cast<float>(i) * spacing;
        float ext = static_cast<float>(halfExtent) * spacing;
        // X 方向线段
        lines.push_back({-ext, 0.0f, fi});
        lines.push_back({ ext, 0.0f, fi});
        // Z 方向线段
        lines.push_back({fi, 0.0f, -ext});
        lines.push_back({fi, 0.0f,  ext});
    }

    m_groundGridVertexCount = static_cast<int>(lines.size());

    glGenVertexArrays(1, &m_groundGridVao);
    glGenBuffers(1, &m_groundGridVbo);
    glBindVertexArray(m_groundGridVao);
    glBindBuffer(GL_ARRAY_BUFFER, m_groundGridVbo);
    glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(glm::vec3), lines.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}

void GLRenderer::drawGroundGrid(const glm::mat4& view, const glm::mat4& projection) {
    glUseProgram(m_wireShader);
    glUniformMatrix4fv(glGetUniformLocation(m_wireShader, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(m_wireShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    glm::vec3 gridColor(0.72f, 0.72f, 0.75f);
    glUniform3fv(glGetUniformLocation(m_wireShader, "wireColor"), 1, glm::value_ptr(gridColor));

    if (m_groundGridVao == 0) {
        buildGroundGrid();  // 只构建一次
    }

    glBindVertexArray(m_groundGridVao);
    glDrawArrays(GL_LINES, 0, m_groundGridVertexCount);
    glBindVertexArray(0);
}

// Factory function
std::unique_ptr<Renderer> createRenderer(const RendererConfig& config) {
    auto renderer = std::make_unique<GLRenderer>();
    renderer->setConfig(config);
    return renderer;
}

} // namespace sim::renderer
