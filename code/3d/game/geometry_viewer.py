import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math


class GeometryRenderer:
    @staticmethod
    def draw_cube():
        vertices = [
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        faces = [
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
            (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)
        ]

        glBegin(GL_QUADS)
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
        for i, face in enumerate(faces):
            glColor3fv(colors[i])
            for vertex in face:
                glVertex3fv(vertices[vertex])
        glEnd()

        glColor3f(0.2, 0.2, 0.2)
        glLineWidth(2)
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()

    @staticmethod
    def draw_box(width=2, height=1.5, depth=1):
        w, h, d = width / 2, height / 2, depth / 2
        vertices = [
            [w, h, -d], [w, -h, -d], [-w, -h, -d], [-w, h, -d],
            [w, h, d], [w, -h, d], [-w, -h, d], [-w, h, d]
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)]

        glBegin(GL_QUADS)
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
        for i, face in enumerate(faces):
            glColor3fv(colors[i])
            for vertex in face:
                glVertex3fv(vertices[vertex])
        glEnd()

        glColor3f(0.2, 0.2, 0.2)
        glLineWidth(2)
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()

    @staticmethod
    def draw_cylinder(radius=1, height=2, slices=32):
        quad = gluNewQuadric()
        glPushMatrix()
        glRotatef(-90, 1, 0, 0)
        glTranslatef(0, 0, -height / 2)

        glColor3f(0.4, 0.6, 0.9)
        gluCylinder(quad, radius, radius, height, slices, 1)

        glColor3f(0.6, 0.8, 1.0)
        gluDisk(quad, 0, radius, slices, 1)

        glTranslatef(0, 0, height)
        glRotatef(180, 1, 0, 0)
        gluDisk(quad, 0, radius, slices, 1)
        glPopMatrix()
        gluDeleteQuadric(quad)

    @staticmethod
    def draw_cone(radius=1, height=2, slices=32):
        quad = gluNewQuadric()
        glPushMatrix()
        glRotatef(-90, 1, 0, 0)
        glTranslatef(0, 0, -height / 2)

        glColor3f(0.9, 0.6, 0.4)
        gluCylinder(quad, radius, 0, height, slices, 1)

        glColor3f(1.0, 0.8, 0.6)
        gluDisk(quad, 0, radius, slices, 1)
        glPopMatrix()
        gluDeleteQuadric(quad)

    @staticmethod
    def draw_sphere(radius=1, slices=32, stacks=32):
        quad = gluNewQuadric()
        glColor3f(0.6, 0.4, 0.9)
        gluSphere(quad, radius, slices, stacks)
        gluDeleteQuadric(quad)


class GeometryObject:
    def __init__(self, geometry_type, position=(0, 0, 0), scale=1.0):
        self.type = geometry_type
        self.position = list(position)
        self.scale = scale
        self.selected = False

    def draw(self):
        glPushMatrix()
        glTranslatef(*self.position)
        glScalef(self.scale, self.scale, self.scale)

        # 如果被选中，绘制高亮边框
        if self.selected:
            glDisable(GL_LIGHTING)
            glColor3f(1, 1, 0)
            glLineWidth(4)
            # 绘制一个立方体边框表示选中
            self.draw_selection_box()
            glEnable(GL_LIGHTING)
            glLineWidth(2)

        # 绘制几何体
        if self.type == 'cube':
            GeometryRenderer.draw_cube()
        elif self.type == 'box':
            GeometryRenderer.draw_box()
        elif self.type == 'cylinder':
            GeometryRenderer.draw_cylinder()
        elif self.type == 'cone':
            GeometryRenderer.draw_cone()
        elif self.type == 'sphere':
            GeometryRenderer.draw_sphere()

        glPopMatrix()

    def draw_selection_box(self):
        size = 2.5
        glBegin(GL_LINE_LOOP)
        glVertex3f(-size, -size, -size)
        glVertex3f(size, -size, -size)
        glVertex3f(size, size, -size)
        glVertex3f(-size, size, -size)
        glEnd()
        glBegin(GL_LINE_LOOP)
        glVertex3f(-size, -size, size)
        glVertex3f(size, -size, size)
        glVertex3f(size, size, size)
        glVertex3f(-size, size, size)
        glEnd()
        glBegin(GL_LINES)
        glVertex3f(-size, -size, -size)
        glVertex3f(-size, -size, size)
        glVertex3f(size, -size, -size)
        glVertex3f(size, -size, size)
        glVertex3f(size, size, -size)
        glVertex3f(size, size, size)
        glVertex3f(-size, size, -size)
        glVertex3f(-size, size, size)
        glEnd()


def setup_3d_viewport(x, y, w, h):
    glViewport(x, y, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, w / h, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)


def setup_ortho_viewport(x, y, w, h):
    glViewport(x, y, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect = w / h
    if aspect > 1:
        glOrtho(-8 * aspect, 8 * aspect, -8, 8, -20, 20)
    else:
        glOrtho(-8, 8, -8 / aspect, 8 / aspect, -20, 20)
    glMatrixMode(GL_MODELVIEW)


def draw_scene_3d(objects, rotation_x, rotation_y):
    glLoadIdentity()
    glTranslatef(0, 0, -20)
    glRotatef(rotation_x, 1, 0, 0)
    glRotatef(rotation_y, 0, 1, 0)

    for obj in objects:
        obj.draw()


def draw_front_view(objects, rotation_x, rotation_y):
    glLoadIdentity()
    glRotatef(rotation_x, 1, 0, 0)
    glRotatef(rotation_y, 0, 1, 0)

    for obj in objects:
        obj.draw()


def draw_top_view(objects, rotation_x, rotation_y):
    glLoadIdentity()
    glRotatef(-90, 1, 0, 0)
    glRotatef(rotation_x, 1, 0, 0)
    glRotatef(rotation_y, 0, 1, 0)

    for obj in objects:
        obj.draw()


def draw_side_view(objects, rotation_x, rotation_y):
    glLoadIdentity()
    glRotatef(-90, 0, 1, 0)
    glRotatef(rotation_x, 1, 0, 0)
    glRotatef(rotation_y, 0, 1, 0)

    for obj in objects:
        obj.draw()


def render_text_to_screen(screen, text, pos, color=(255, 255, 255), size=28):
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, pos)


def main():
    pygame.init()
    width, height = 1400, 900

    screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption('几何体自由组装三视图生成器')

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    rotation_x = 20
    rotation_y = 30
    mouse_down = False
    last_pos = (0, 0)

    geometries = ['cube', 'box', 'cylinder', 'cone', 'sphere']
    current_geometry = 0

    # 几何体列表
    objects = []
    selected_object = None
    dragging_object = False  # 是否在拖拽物体

    clock = pygame.time.Clock()
    text_overlay = pygame.Surface((width, height), pygame.SRCALPHA)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键
                    pos = pygame.mouse.get_pos()
                    # 只在3D视图区域响应鼠标
                    if pos[0] < width // 2 and pos[1] < height // 2:
                        # 如果有选中的物体，优先拖拽物体
                        if selected_object:
                            dragging_object = True
                            last_pos = pos
                        else:
                            mouse_down = True
                            last_pos = pos

            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down = False
                    dragging_object = False

            elif event.type == MOUSEWHEEL:
                # 鼠标滚轮调整选中物体大小
                if selected_object:
                    if event.y > 0:  # 向上滚
                        selected_object.scale *= 1.1
                    else:  # 向下滚
                        selected_object.scale *= 0.9
                    selected_object.scale = max(0.1, min(selected_object.scale, 5.0))

            elif event.type == MOUSEMOTION:
                if dragging_object and selected_object:
                    # 拖拽物体
                    pos = pygame.mouse.get_pos()
                    dx = pos[0] - last_pos[0]
                    dy = pos[1] - last_pos[1]
                    # 根据当前视角旋转计算移动方向
                    move_scale = 0.03

                    # 将屏幕空间的移动转换为世界空间
                    # 考虑旋转角度
                    angle_y_rad = math.radians(rotation_y)
                    angle_x_rad = math.radians(rotation_x)

                    # X轴移动（左右）
                    selected_object.position[0] += dx * move_scale * math.cos(angle_y_rad)
                    selected_object.position[2] += dx * move_scale * math.sin(angle_y_rad)

                    # Y轴移动（上下）
                    selected_object.position[1] -= dy * move_scale * math.cos(angle_x_rad)
                    selected_object.position[2] -= dy * move_scale * math.sin(angle_x_rad) * math.sin(angle_y_rad)

                    last_pos = pos
                elif mouse_down:
                    # 旋转视角
                    pos = pygame.mouse.get_pos()
                    dx = pos[0] - last_pos[0]
                    dy = pos[1] - last_pos[1]
                    rotation_y += dx * 0.5
                    rotation_x += dy * 0.5
                    last_pos = pos

            elif event.type == KEYDOWN:
                if event.key == K_SPACE:  # 切换几何体类型
                    current_geometry = (current_geometry + 1) % len(geometries)

                elif event.key == K_a:  # 添加几何体并自动选中
                    new_obj = GeometryObject(geometries[current_geometry], (0, 0, 0), 1.0)
                    objects.append(new_obj)
                    # 取消之前的选中
                    if selected_object:
                        selected_object.selected = False
                    # 选中新添加的物体
                    selected_object = new_obj
                    selected_object.selected = True

                elif event.key == K_d:  # 删除选中的几何体
                    if selected_object and selected_object in objects:
                        objects.remove(selected_object)
                        selected_object = None

                elif event.key == K_s:  # 选择下一个几何体
                    if objects:
                        if selected_object:
                            selected_object.selected = False
                            idx = objects.index(selected_object)
                            idx = (idx + 1) % len(objects)
                            selected_object = objects[idx]
                        else:
                            selected_object = objects[0]
                        selected_object.selected = True

                elif event.key == K_ESCAPE:  # ESC键取消选中
                    if selected_object:
                        selected_object.selected = False
                        selected_object = None

                elif event.key == K_c:  # 清空所有几何体
                    objects.clear()
                    selected_object = None

                elif event.key == K_r:  # 重置视角
                    rotation_x = 20
                    rotation_y = 30

                # 移动选中的几何体
                elif event.key == K_UP and selected_object:
                    selected_object.position[1] += 0.5
                elif event.key == K_DOWN and selected_object:
                    selected_object.position[1] -= 0.5
                elif event.key == K_LEFT and selected_object:
                    selected_object.position[0] -= 0.5
                elif event.key == K_RIGHT and selected_object:
                    selected_object.position[0] += 0.5
                elif event.key == K_PAGEUP and selected_object:
                    selected_object.position[2] += 0.5
                elif event.key == K_PAGEDOWN and selected_object:
                    selected_object.position[2] -= 0.5

                # 缩放选中的几何体
                elif event.key == K_EQUALS and selected_object:  # + 键
                    selected_object.scale *= 1.1
                elif event.key == K_MINUS and selected_object:
                    selected_object.scale *= 0.9

        # 清除缓冲
        glClearColor(0.15, 0.15, 0.2, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 3D视图
        setup_3d_viewport(0, height // 2, width // 2, height // 2)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightfv(GL_LIGHT0, GL_POSITION, [5, 5, 5, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.6, 0.6, 0.6, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        draw_scene_3d(objects, rotation_x, rotation_y)

        #主视图
        setup_ortho_viewport(width // 2, height // 2, width // 2, height // 2)
        glDisable(GL_LIGHTING)
        draw_front_view(objects, rotation_x, rotation_y)

        #俯视图
        setup_ortho_viewport(0, 0, width // 2, height // 2)
        draw_top_view(objects, rotation_x, rotation_y)

        #侧视图
        setup_ortho_viewport(width // 2, 0, width // 2, height // 2)
        draw_side_view(objects, rotation_x, rotation_y)

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, width, 0, height)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glColor3f(0.9, 0.9, 0.9)
        glLineWidth(3)
        glBegin(GL_LINES)
        glVertex2f(width // 2, 0)
        glVertex2f(width // 2, height)
        glVertex2f(0, height // 2)
        glVertex2f(width, height // 2)
        glEnd()

        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        text_overlay.fill((0, 0, 0, 0))

        # 视图标题
        render_text_to_screen(text_overlay, '3D View (Drag to Rotate)', (20, 20), (255, 255, 100), 26)
        render_text_to_screen(text_overlay, 'Front View', (width // 2 + 20, 20), (100, 255, 100), 26)
        render_text_to_screen(text_overlay, 'Top View', (20, height // 2 + 20), (100, 200, 255), 26)
        render_text_to_screen(text_overlay, 'Right Side View', (width // 2 + 20, height // 2 + 20), (255, 150, 150), 26)

        # 控制说明
        y_offset = height - 200
        render_text_to_screen(text_overlay,
                              f'Current: {geometries[current_geometry].upper()}  |  Objects: {len(objects)}',
                              (20, y_offset), (255, 255, 255), 28)
        y_offset += 30
        render_text_to_screen(text_overlay,
                              'SPACE: Switch Type  |  A: Add  |  S: Select Next  |  D: Delete  |  C: Clear',
                              (20, y_offset), (220, 220, 220), 22)
        y_offset += 25
        render_text_to_screen(text_overlay,
                              'ESC: Deselect',
                              (20, y_offset), (220, 220, 220), 22)

        if selected_object:
            y_offset += 30
            render_text_to_screen(text_overlay,
                                  f'Selected: Pos({selected_object.position[0]:.1f}, {selected_object.position[1]:.1f}, {selected_object.position[2]:.1f})  Scale: {selected_object.scale:.2f}',
                                  (20, y_offset), (255, 255, 100), 22)

        text_data = pygame.image.tostring(text_overlay, "RGBA", True)
        glWindowPos2d(0, 0)
        glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()

    