// main.cpp
// 简化版：鼠标旋转 + 实时三视图
// 编译: g++ main.cpp -o geometry_viewer -lglfw -lGL -lGLEW -lGLU -std=c++17

#define _USE_MATH_DEFINES
#include <cmath>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 窗口尺寸
const int WINDOW_WIDTH = 1400;
const int WINDOW_HEIGHT = 900;

// 几何体类型
enum ShapeType {
    CUBE,
    CUBOID,
    CYLINDER,
    CONE,
    PYRAMID,
    SPHERE
};

// 全局变量
ShapeType currentShape = CUBE;
float rotationX = 30.0f;
float rotationY = 45.0f;

// 鼠标控制
bool mousePressed = false;
double lastMouseX = 0.0;
double lastMouseY = 0.0;

// 顶点着色器
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

// 片段着色器
const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 objectColor;
uniform vec3 lightPos;
uniform vec3 viewPos;

void main() {
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * vec3(1.0);
    
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0);
    
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * vec3(1.0);
    
    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
)";

// 2D线条着色器
const char* lineVertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
uniform mat4 projection;
void main() {
    gl_Position = projection * vec4(aPos, 0.0, 1.0);
}
)";

const char* lineFragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
uniform vec3 lineColor;
void main() {
    FragColor = vec4(lineColor, 1.0);
}
)";

// 错误检查
void checkGLError(const char* location) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "OpenGL错误 [" << location << "]: " << err << std::endl;
    }
}

// 编译着色器
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "着色器编译失败: " << infoLog << std::endl;
    }
    return shader;
}

// 创建着色器程序
GLuint createShaderProgram(const char* vertexSrc, const char* fragmentSrc) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSrc);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "程序链接失败: " << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

// 生成正方体
void generateCube(std::vector<float>& vertices, std::vector<unsigned int>& indices) {
    vertices = {
        // 前面 (Z+)
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
         0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
        // 后面 (Z-)
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
         0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
        // 顶面 (Y+)
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
        // 底面 (Y-)
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
         0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
        // 右面 (X+)
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
         // 左面 (X-)
         -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
         -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
         -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
         -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
    };

    indices = {
        0, 1, 2, 2, 3, 0,
        4, 6, 5, 6, 4, 7,
        8, 9, 10, 10, 11, 8,
        12, 14, 13, 14, 12, 15,
        16, 17, 18, 18, 19, 16,
        20, 22, 21, 22, 20, 23
    };
}

// 生成长方体
void generateCuboid(std::vector<float>& vertices, std::vector<unsigned int>& indices) {
    generateCube(vertices, indices);
    for (size_t i = 0; i < vertices.size(); i += 6) {
        vertices[i] *= 1.5f;      // X拉伸
        vertices[i + 1] *= 0.7f;    // Y压缩
        vertices[i + 2] *= 0.9f;    // Z稍微压缩
    }
}

// 生成圆柱
void generateCylinder(std::vector<float>& vertices, std::vector<unsigned int>& indices, int segments = 36) {
    vertices.clear();
    indices.clear();

    float radius = 0.5f;
    float height = 1.0f;

    // 顶面和底面圆心
    vertices.insert(vertices.end(), { 0.0f, height / 2, 0.0f, 0.0f, 1.0f, 0.0f });
    vertices.insert(vertices.end(), { 0.0f, -height / 2, 0.0f, 0.0f, -1.0f, 0.0f });

    // 顶面和底面圆周
    for (int i = 0; i <= segments; ++i) {
        float angle = 2.0f * M_PI * i / segments;
        float x = radius * cos(angle);
        float z = radius * sin(angle);

        vertices.insert(vertices.end(), { x, height / 2, z, 0.0f, 1.0f, 0.0f });
        vertices.insert(vertices.end(), { x, -height / 2, z, 0.0f, -1.0f, 0.0f });

        float nx = cos(angle);
        float nz = sin(angle);
        vertices.insert(vertices.end(), { x, height / 2, z, nx, 0.0f, nz });
        vertices.insert(vertices.end(), { x, -height / 2, z, nx, 0.0f, nz });
    }

    // 顶面索引
    for (int i = 0; i < segments; ++i) {
        indices.push_back(0);
        indices.push_back(2 + i * 2);
        indices.push_back(2 + (i + 1) * 2);
    }

    // 底面索引
    for (int i = 0; i < segments; ++i) {
        indices.push_back(1);
        indices.push_back(3 + (i + 1) * 2);
        indices.push_back(3 + i * 2);
    }

    // 侧面索引
    int sideStart = 2 + 2 * (segments + 1);
    for (int i = 0; i < segments; ++i) {
        int current = sideStart + i * 2;
        int next = sideStart + (i + 1) * 2;

        indices.push_back(current);
        indices.push_back(current + 1);
        indices.push_back(next);

        indices.push_back(next);
        indices.push_back(current + 1);
        indices.push_back(next + 1);
    }
}

// 生成圆锥
void generateCone(std::vector<float>& vertices, std::vector<unsigned int>& indices, int segments = 36) {
    vertices.clear();
    indices.clear();

    float radius = 0.5f;
    float height = 1.0f;

    // 顶点
    vertices.insert(vertices.end(), { 0.0f, height, 0.0f, 0.0f, 1.0f, 0.0f });

    // 底面中心
    vertices.insert(vertices.end(), { 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f });

    // 底面圆周和侧面
    for (int i = 0; i <= segments; ++i) {
        float angle = 2.0f * M_PI * i / segments;
        float x = radius * cos(angle);
        float z = radius * sin(angle);

        // 底面顶点
        vertices.insert(vertices.end(), { x, 0.0f, z, 0.0f, -1.0f, 0.0f });

        // 侧面法向量
        float slant = sqrt(radius * radius + height * height);
        float nx = cos(angle) * height / slant;
        float ny = radius / slant;
        float nz = sin(angle) * height / slant;

        vertices.insert(vertices.end(), { 0.0f, height, 0.0f, nx, ny, nz });
        vertices.insert(vertices.end(), { x, 0.0f, z, nx, ny, nz });
    }

    // 底面索引
    for (int i = 0; i < segments; ++i) {
        indices.push_back(1);
        indices.push_back(2 + i);
        indices.push_back(2 + (i + 1));
    }

    // 侧面索引
    int sideStart = 2 + segments + 1;
    for (int i = 0; i < segments; ++i) {
        int current = sideStart + i * 2;
        indices.push_back(current);
        indices.push_back(current + 1);
        indices.push_back(current + 3);
    }
}

// 生成金字塔
void generatePyramid(std::vector<float>& vertices, std::vector<unsigned int>& indices) {
    vertices.clear();
    indices.clear();

    glm::vec3 apex(0.0f, 0.75f, 0.0f);
    glm::vec3 base[4] = {
        glm::vec3(-0.5f, 0.0f, -0.5f),
        glm::vec3(0.5f, 0.0f, -0.5f),
        glm::vec3(0.5f, 0.0f,  0.5f),
        glm::vec3(-0.5f, 0.0f,  0.5f)
    };

    // 底面（向下的法向量）
    glm::vec3 bottomNormal(0.0f, -1.0f, 0.0f);
    for (int i = 0; i < 4; ++i) {
        vertices.insert(vertices.end(), { base[i].x, base[i].y, base[i].z,
                                         bottomNormal.x, bottomNormal.y, bottomNormal.z });
    }

    // 四个侧面
    for (int i = 0; i < 4; ++i) {
        int next = (i + 1) % 4;
        glm::vec3 v1 = base[i] - apex;
        glm::vec3 v2 = base[next] - apex;
        glm::vec3 normal = glm::normalize(glm::cross(v2, v1));

        vertices.insert(vertices.end(), { apex.x, apex.y, apex.z, normal.x, normal.y, normal.z });
        vertices.insert(vertices.end(), { base[i].x, base[i].y, base[i].z, normal.x, normal.y, normal.z });
        vertices.insert(vertices.end(), { base[next].x, base[next].y, base[next].z, normal.x, normal.y, normal.z });
    }

    // 底面索引
    indices = { 0, 1, 2, 2, 3, 0 };

    // 侧面索引
    for (unsigned int i = 0; i < 4; ++i) {
        unsigned int start = 4 + i * 3;
        indices.push_back(start);
        indices.push_back(start + 1);
        indices.push_back(start + 2);
    }
}

// 生成球体
void generateSphere(std::vector<float>& vertices, std::vector<unsigned int>& indices, int sectors = 36, int stacks = 18) {
    vertices.clear();
    indices.clear();

    float radius = 0.5f;

    for (int i = 0; i <= stacks; ++i) {
        float phi = M_PI * i / stacks;
        float y = radius * cos(phi);
        float r = radius * sin(phi);

        for (int j = 0; j <= sectors; ++j) {
            float theta = 2.0f * M_PI * j / sectors;
            float x = r * cos(theta);
            float z = r * sin(theta);

            glm::vec3 normal = glm::normalize(glm::vec3(x, y, z));
            vertices.insert(vertices.end(), { x, y, z, normal.x, normal.y, normal.z });
        }
    }

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < sectors; ++j) {
            unsigned int first = i * (sectors + 1) + j;
            unsigned int second = first + sectors + 1;

            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }
}

// 绘制2D投影视图
void draw2DView(GLuint program, const std::vector<float>& vertices,
    int viewType, float x, float y, float w, float h) {
    std::vector<float> points;
    std::vector<unsigned int> lineIndices;

    // 将3D顶点投影到2D
    for (size_t i = 0; i < vertices.size(); i += 6) {
        float px = vertices[i];
        float py = vertices[i + 1];
        float pz = vertices[i + 2];

        float x2d, y2d;
        switch (viewType) {
        case 0: // 正视图 (YZ平面)
            x2d = pz; y2d = py;
            break;
        case 1: // 侧视图 (XY平面)
            x2d = px; y2d = py;
            break;
        case 2: // 俯视图 (XZ平面)
            x2d = px; y2d = pz;
            break;
        }

        points.push_back(x + w / 2 + x2d * w * 0.7f);
        points.push_back(y + h / 2 + y2d * h * 0.7f);
    }

    // 生成线段（简化：连接相邻顶点）
    for (size_t i = 0; i < points.size() / 2 - 1; ++i) {
        lineIndices.push_back(i);
        lineIndices.push_back(i + 1);
    }

    if (points.empty() || lineIndices.empty()) return;

    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), points.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, lineIndices.size() * sizeof(unsigned int),
        lineIndices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glUseProgram(program);
    glm::mat4 projection = glm::ortho(0.0f, (float)WINDOW_WIDTH, 0.0f, (float)WINDOW_HEIGHT);
    glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    glm::vec3 color;
    switch (viewType) {
    case 0: color = glm::vec3(0.2f, 0.8f, 0.4f); break; // 绿色
    case 1: color = glm::vec3(1.0f, 0.6f, 0.2f); break; // 橙色
    case 2: color = glm::vec3(0.6f, 0.4f, 1.0f); break; // 紫色
    }
    glUniform3fv(glGetUniformLocation(program, "lineColor"), 1, glm::value_ptr(color));

    glLineWidth(2.0f);
    glDrawElements(GL_LINES, lineIndices.size(), GL_UNSIGNED_INT, 0);

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
}

// 鼠标按钮回调
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mousePressed = true;
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
        }
        else if (action == GLFW_RELEASE) {
            mousePressed = false;
        }
    }
}

// 鼠标移动回调
void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    if (mousePressed) {
        float deltaX = (float)(xpos - lastMouseX);
        float deltaY = (float)(ypos - lastMouseY);

        rotationY += deltaX * 0.5f;
        rotationX += deltaY * 0.5f;

        // 限制X轴旋转范围
        if (rotationX > 89.0f) rotationX = 89.0f;
        if (rotationX < -89.0f) rotationX = -89.0f;

        lastMouseX = xpos;
        lastMouseY = ypos;
    }
}

// 键盘回调
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        switch (key) {
        case GLFW_KEY_1: currentShape = CUBE; break;
        case GLFW_KEY_2: currentShape = CUBOID; break;
        case GLFW_KEY_3: currentShape = CYLINDER; break;
        case GLFW_KEY_4: currentShape = CONE; break;
        case GLFW_KEY_5: currentShape = PYRAMID; break;
        case GLFW_KEY_6: currentShape = SPHERE; break;
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, true); break;
        case GLFW_KEY_R: // 重置视角
            rotationX = 30.0f;
            rotationY = 45.0f;
            break;
        }
    }
}

int main() {
    // 初始化GLFW
    if (!glfwInit()) {
        std::cerr << "GLFW初始化失败" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT,
        "OpenGL 几何体三视图 - 鼠标拖动旋转", nullptr, nullptr);
    if (!window) {
        std::cerr << "窗口创建失败" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);

    // 初始化GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW初始化失败" << std::endl;
        return -1;
    }

    glGetError(); // 清除GLEW初始化错误

    std::cout << "========================================" << std::endl;
    std::cout << "OpenGL版本: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "显卡: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\n控制说明：" << std::endl;
    std::cout << "  鼠标左键拖动 - 旋转视角" << std::endl;
    std::cout << "  1 - 正方体" << std::endl;
    std::cout << "  2 - 长方体" << std::endl;
    std::cout << "  3 - 圆柱体" << std::endl;
    std::cout << "  4 - 圆锥" << std::endl;
    std::cout << "  5 - 四棱锥" << std::endl;
    std::cout << "  6 - 球体" << std::endl;
    std::cout << "  R - 重置视角" << std::endl;
    std::cout << "  ESC - 退出" << std::endl;
    std::cout << "========================================\n" << std::endl;

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLuint shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    GLuint lineProgram = createShaderProgram(lineVertexShaderSource, lineFragmentShaderSource);

    // 主循环
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 生成几何体
        std::vector<float> vertices;
        std::vector<unsigned int> indices;

        switch (currentShape) {
        case CUBE: generateCube(vertices, indices); break;
        case CUBOID: generateCuboid(vertices, indices); break;
        case CYLINDER: generateCylinder(vertices, indices); break;
        case CONE: generateCone(vertices, indices); break;
        case PYRAMID: generatePyramid(vertices, indices); break;
        case SPHERE: generateSphere(vertices, indices); break;
        }

        // 创建VAO/VBO/EBO
        GLuint VAO, VBO, EBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // ========== 绘制主3D视图（左侧2/3） ==========
        glViewport(0, 0, WINDOW_WIDTH * 2 / 3, WINDOW_HEIGHT);
        glEnable(GL_DEPTH_TEST);

        glUseProgram(shaderProgram);

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(model, glm::radians(rotationX), glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, glm::radians(rotationY), glm::vec3(0.0f, 1.0f, 0.0f));

        glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f));

        float aspect = (WINDOW_WIDTH * 2.0f / 3.0f) / WINDOW_HEIGHT;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 0.3f, 0.6f, 0.95f);
        glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), 2.0f, 2.0f, 2.0f);
        glUniform3f(glGetUniformLocation(shaderProgram, "viewPos"), 0.0f, 0.0f, 3.0f);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

        // ========== 绘制三视图（右侧1/3，分三栏） ==========
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        glDisable(GL_DEPTH_TEST);

        float rightX = WINDOW_WIDTH * 2.0f / 3.0f;
        float viewWidth = WINDOW_WIDTH / 3.0f;
        float viewHeight = WINDOW_HEIGHT / 3.0f;

        // 正视图（右上）
        draw2DView(lineProgram, vertices, 0, rightX, WINDOW_HEIGHT * 2.0f / 3.0f, viewWidth, viewHeight);

        // 侧视图（右中）
        draw2DView(lineProgram, vertices, 1, rightX, WINDOW_HEIGHT / 3.0f, viewWidth, viewHeight);

        // 俯视图（右下）
        draw2DView(lineProgram, vertices, 2, rightX, 0, viewWidth, viewHeight);

        // 清理
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteProgram(shaderProgram);
    glDeleteProgram(lineProgram);
    glfwTerminate();
    return 0;
}