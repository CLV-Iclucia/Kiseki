#include "gl-render-backend.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

namespace ksk::renderer {

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

    vec3 calcBlinnPhong(vec3 lightDir, vec3 viewDir, vec3 norm,
                        vec3 baseColor, vec3 lightColor, float intensity) {
        float NdotL = dot(norm, lightDir);
        float diff = max(NdotL, 0.0);
        float wrapDiff = diff * 0.8 + 0.2;
        vec3 diffuse = wrapDiff * baseColor * lightColor * intensity;

        vec3 halfDir = normalize(lightDir + viewDir);
        float NdotH = max(dot(norm, halfDir), 0.0);
        float spec = pow(NdotH, 32.0);
        spec *= step(0.0, NdotL);
        vec3 specular = spec * lightColor * intensity * 0.5;

        return diffuse + specular;
    }

    void main() {
        vec3 baseColor = (dot(VertexColor, VertexColor) > 0.001)
                         ? VertexColor : objectColor;

        vec3 norm = normalize(Normal);
        vec3 viewDir = normalize(viewPos - FragPos);

        float upFactor = dot(norm, vec3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
        vec3 ambientColor = mix(vec3(0.08, 0.10, 0.14),
                                vec3(0.18, 0.16, 0.14),
                                upFactor);
        vec3 ambient = ambientColor * baseColor;

        vec3 lightDir1 = normalize(lightPos - FragPos);
        vec3 key = calcBlinnPhong(lightDir1, viewDir, norm,
                                  baseColor, vec3(1.0, 0.95, 0.9), 1.0);

        vec3 lightDir2 = normalize(lightPos2 - FragPos);
        vec3 fill = calcBlinnPhong(lightDir2, viewDir, norm,
                                   baseColor, vec3(0.85, 0.9, 1.0), 0.35);

        float fresnel = 1.0 - max(dot(norm, viewDir), 0.0);
        fresnel = pow(fresnel, 3.0) * 0.25;
        vec3 rim = fresnel * vec3(0.6, 0.7, 0.8);

        vec3 result = ambient + key + fill + rim;
        result = pow(result, vec3(1.0 / 2.2));

        FragColor = vec4(result, 1.0);
    }
)";

static const char* particleVertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;

    uniform mat4 view;
    uniform mat4 projection;
    uniform float pointSize;
    uniform float viewportHeight;

    void main() {
        vec4 viewPos = view * vec4(aPos, 1.0);
        gl_Position = projection * viewPos;
        float dist = -viewPos.z;
        gl_PointSize = max(1.0, pointSize * projection[1][1] * viewportHeight / dist);
    }
)";

static const char* particleFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    uniform vec3 particleColor;

    void main() {
        vec2 coord = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(coord, coord);
        if (r2 > 1.0) discard;

        float shade = 1.0 - r2 * 0.5;
        FragColor = vec4(particleColor * shade, 1.0);
    }
)";

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

static unsigned int compileShader(unsigned int type, const char* source) {
  unsigned int shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);

  int success = 0;
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

static unsigned int createShaderProgram(const char* vertexSrc,
                                        const char* fragmentSrc) {
  unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexSrc);
  unsigned int fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);

  unsigned int program = glCreateProgram();
  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);
  glLinkProgram(program);

  int success = 0;
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

void GLRenderBackend::configureWindow() {
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
}

bool GLRenderBackend::initialize(const RenderBackendInit& init) {
  m_window = static_cast<GLFWwindow*>(init.nativeWindow);
  if (!m_window) {
    std::cerr << "GLRenderBackend requires a GLFW window" << std::endl;
    return false;
  }

  glfwMakeContextCurrent(m_window);
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to load OpenGL functions" << std::endl;
    return false;
  }

  glfwSwapInterval(init.vsync ? 1 : 0);

  m_meshShader = createShaderProgram(vertexShaderSource, fragmentShaderSource);
  m_wireShader = createShaderProgram(wireVertexShaderSource, wireFragmentShaderSource);
  m_particleShader = createShaderProgram(particleVertexShaderSource,
                                         particleFragmentShaderSource);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_MULTISAMPLE);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glClearColor(0.92f, 0.92f, 0.94f, 1.0f);
  return m_meshShader != 0 && m_wireShader != 0 && m_particleShader != 0;
}

void GLRenderBackend::renderFrame(const SceneProxy& scene,
                                  const RenderFrameContext& context) {
  glViewport(0, 0, context.framebufferWidth, context.framebufferHeight);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glUseProgram(m_meshShader);

  glm::mat4 model = glm::mat4(1.0f);
  glUniformMatrix4fv(glGetUniformLocation(m_meshShader, "model"), 1, GL_FALSE,
                     glm::value_ptr(model));
  glUniformMatrix4fv(glGetUniformLocation(m_meshShader, "view"), 1, GL_FALSE,
                     glm::value_ptr(context.view));
  glUniformMatrix4fv(glGetUniformLocation(m_meshShader, "projection"), 1,
                     GL_FALSE, glm::value_ptr(context.projection));

  float lightScale = std::max(context.cameraDistance * 0.6f, 3.0f);
  glm::vec3 lightPos = context.cameraTarget +
      glm::vec3(lightScale * 0.6f, lightScale, lightScale * 0.8f);
  glUniform3fv(glGetUniformLocation(m_meshShader, "lightPos"), 1,
               glm::value_ptr(lightPos));

  glm::vec3 lightPos2 = context.cameraTarget +
      glm::vec3(-lightScale * 0.5f, lightScale * 0.3f, -lightScale * 0.7f);
  glUniform3fv(glGetUniformLocation(m_meshShader, "lightPos2"), 1,
               glm::value_ptr(lightPos2));

  glUniform3fv(glGetUniformLocation(m_meshShader, "viewPos"), 1,
               glm::value_ptr(context.cameraPosition));

  glm::vec3 defaultColor(0.75f, 0.55f, 0.38f);
  glUniform3fv(glGetUniformLocation(m_meshShader, "objectColor"), 1,
               glm::value_ptr(defaultColor));

  for (const auto& mesh : scene.meshes) {
    drawMesh(mesh);
  }

  if (!scene.wireframes.empty()) {
    glUseProgram(m_wireShader);
    glUniformMatrix4fv(glGetUniformLocation(m_wireShader, "view"), 1,
                       GL_FALSE, glm::value_ptr(context.view));
    glUniformMatrix4fv(glGetUniformLocation(m_wireShader, "projection"), 1,
                       GL_FALSE, glm::value_ptr(context.projection));

    for (const auto& wire : scene.wireframes) {
      drawWireframe(wire);
    }
  }

  for (const auto& particle : scene.particles) {
    drawParticles(particle, context);
  }

  drawGroundGrid(context);
}

void GLRenderBackend::cleanup() {
  for (auto& [name, state] : m_meshCache) {
    glDeleteVertexArrays(1, &state.vao);
    glDeleteBuffers(1, &state.vbo);
    glDeleteBuffers(1, &state.ebo);
    if (state.nbo) glDeleteBuffers(1, &state.nbo);
  }
  m_meshCache.clear();

  if (m_groundGridVao) glDeleteVertexArrays(1, &m_groundGridVao);
  if (m_groundGridVbo) glDeleteBuffers(1, &m_groundGridVbo);
  m_groundGridVao = 0;
  m_groundGridVbo = 0;
  m_groundGridVertexCount = 0;

  if (m_meshShader) glDeleteProgram(m_meshShader);
  if (m_wireShader) glDeleteProgram(m_wireShader);
  if (m_particleShader) glDeleteProgram(m_particleShader);
  m_meshShader = 0;
  m_wireShader = 0;
  m_particleShader = 0;
  m_window = nullptr;
}

bool GLRenderBackend::present() {
  if (!m_window) return false;
  glfwSwapBuffers(m_window);
  return true;
}

void GLRenderBackend::uploadMesh(const MeshProxy& mesh) {
  auto it = m_meshCache.find(mesh.name);
  bool needCreate = (it == m_meshCache.end());

  GLMeshState state;
  if (!needCreate) {
    state = it->second;
    if (state.vertexCount != mesh.positions.size()) {
      glDeleteVertexArrays(1, &state.vao);
      glDeleteBuffers(1, &state.vbo);
      glDeleteBuffers(1, &state.ebo);
      if (state.nbo) glDeleteBuffers(1, &state.nbo);
      needCreate = true;
    }
  }

  if (needCreate) {
    glGenVertexArrays(1, &state.vao);
    glGenBuffers(1, &state.vbo);
    glGenBuffers(1, &state.ebo);
    glGenBuffers(1, &state.nbo);
    state.vertexCount = mesh.positions.size();
  }

  state.indexCount = static_cast<int>(mesh.triangles.size() * 3);

  glBindVertexArray(state.vao);

  glBindBuffer(GL_ARRAY_BUFFER, state.vbo);
  glBufferData(GL_ARRAY_BUFFER, mesh.positions.size() * sizeof(core::Vec3f),
               mesh.positions.data(), GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(core::Vec3f),
                        nullptr);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, state.nbo);
  if (!mesh.normals.empty()) {
    glBufferData(GL_ARRAY_BUFFER, mesh.normals.size() * sizeof(core::Vec3f),
                 mesh.normals.data(), GL_DYNAMIC_DRAW);
  } else {
    std::vector<core::Vec3f> defaultNormals(mesh.positions.size(),
                                            core::Vec3f(0.0f, 1.0f, 0.0f));
    glBufferData(GL_ARRAY_BUFFER,
                 defaultNormals.size() * sizeof(core::Vec3f),
                 defaultNormals.data(), GL_DYNAMIC_DRAW);
  }
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(core::Vec3f),
                        nullptr);
  glEnableVertexAttribArray(1);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, state.ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,
               mesh.triangles.size() * sizeof(core::Vec3u),
               mesh.triangles.data(), GL_DYNAMIC_DRAW);

  glBindVertexArray(0);

  m_meshCache[mesh.name] = state;
}

void GLRenderBackend::drawMesh(const MeshProxy& mesh) {
  uploadMesh(mesh);

  auto it = m_meshCache.find(mesh.name);
  if (it == m_meshCache.end()) return;

  const auto& state = it->second;

  glUseProgram(m_meshShader);

  if (mesh.objectColor.x >= 0.0f) {
    glUniform3fv(glGetUniformLocation(m_meshShader, "objectColor"), 1,
                 glm::value_ptr(mesh.objectColor));
  }

  glBindVertexArray(state.vao);
  glDrawElements(GL_TRIANGLES, state.indexCount, GL_UNSIGNED_INT, nullptr);
  glBindVertexArray(0);
}

void GLRenderBackend::drawWireframe(const WireframeProxy& wf) {
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
               static_cast<GLsizeiptr>(wf.positions.size() *
                                        sizeof(core::Vec3f)),
               wf.positions.data(), GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(core::Vec3f),
                        nullptr);
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

void GLRenderBackend::drawParticles(const ParticleProxy& particles,
                                    const RenderFrameContext& context) {
  if (particles.positions.empty() || !m_particleShader) return;

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
               static_cast<GLsizeiptr>(particles.positions.size() *
                                        sizeof(core::Vec3f)),
               particles.positions.data(), GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(core::Vec3f),
                        nullptr);
  glEnableVertexAttribArray(0);
  glBindVertexArray(0);

  m_meshCache[particles.name] = state;

  glUseProgram(m_particleShader);

  glUniformMatrix4fv(glGetUniformLocation(m_particleShader, "view"), 1,
                     GL_FALSE, glm::value_ptr(context.view));
  glUniformMatrix4fv(glGetUniformLocation(m_particleShader, "projection"), 1,
                     GL_FALSE, glm::value_ptr(context.projection));

  glUniform1f(glGetUniformLocation(m_particleShader, "pointSize"),
              particles.radius);
  glUniform1f(glGetUniformLocation(m_particleShader, "viewportHeight"),
              static_cast<float>(std::max(context.framebufferHeight, 1)));
  glUniform3fv(glGetUniformLocation(m_particleShader, "particleColor"), 1,
               &particles.color.x);

  glBindVertexArray(state.vao);
  glDrawArrays(GL_POINTS, 0,
               static_cast<GLsizei>(particles.positions.size()));
  glBindVertexArray(0);
}

void GLRenderBackend::buildGroundGrid() {
  const int halfExtent = 5;
  const float spacing = 0.5f;
  std::vector<glm::vec3> lines;

  for (int i = -halfExtent; i <= halfExtent; i++) {
    float fi = static_cast<float>(i) * spacing;
    float ext = static_cast<float>(halfExtent) * spacing;
    lines.push_back({-ext, 0.0f, fi});
    lines.push_back({ ext, 0.0f, fi});
    lines.push_back({fi, 0.0f, -ext});
    lines.push_back({fi, 0.0f,  ext});
  }

  m_groundGridVertexCount = static_cast<int>(lines.size());

  glGenVertexArrays(1, &m_groundGridVao);
  glGenBuffers(1, &m_groundGridVbo);
  glBindVertexArray(m_groundGridVao);
  glBindBuffer(GL_ARRAY_BUFFER, m_groundGridVbo);
  glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(glm::vec3),
               lines.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
  glEnableVertexAttribArray(0);
  glBindVertexArray(0);
}

void GLRenderBackend::drawGroundGrid(const RenderFrameContext& context) {
  glUseProgram(m_wireShader);
  glUniformMatrix4fv(glGetUniformLocation(m_wireShader, "view"), 1,
                     GL_FALSE, glm::value_ptr(context.view));
  glUniformMatrix4fv(glGetUniformLocation(m_wireShader, "projection"), 1,
                     GL_FALSE, glm::value_ptr(context.projection));

  glm::vec3 gridColor(0.72f, 0.72f, 0.75f);
  glUniform3fv(glGetUniformLocation(m_wireShader, "wireColor"), 1,
               glm::value_ptr(gridColor));

  if (m_groundGridVao == 0) {
    buildGroundGrid();
  }

  glBindVertexArray(m_groundGridVao);
  glDrawArrays(GL_LINES, 0, m_groundGridVertexCount);
  glBindVertexArray(0);
}

} // namespace ksk::renderer
