import moderngl
import numpy as np
import trimesh
import pygame
import sys
from pygame.locals import *

class GLBRenderer:
    def __init__(self, model_path, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)

        self.ctx = moderngl.create_context()
        print(f"OpenGL Version: {self.ctx.version_code}")

        # Load and prepare model
        scene = trimesh.load(model_path)
        if isinstance(scene, trimesh.Scene):
            mesh = next(iter(scene.geometry.values()))
        else:
            mesh = scene

        # Create debug cube vertices (position and color)
        self.debug_vertices = np.array([
            # front face (red)
            -0.5, -0.5,  0.5,  1.0, 0.0, 0.0,
             0.5, -0.5,  0.5,  1.0, 0.0, 0.0,
             0.5,  0.5,  0.5,  1.0, 0.0, 0.0,
            -0.5,  0.5,  0.5,  1.0, 0.0, 0.0,
            # back face (green)
            -0.5, -0.5, -0.5,  0.0, 1.0, 0.0,
             0.5, -0.5, -0.5,  0.0, 1.0, 0.0,
             0.5,  0.5, -0.5,  0.0, 1.0, 0.0,
            -0.5,  0.5, -0.5,  0.0, 1.0, 0.0,
        ], dtype='f4')

        self.debug_indices = np.array([
            0, 1, 2, 2, 3, 0,  # front
            1, 5, 6, 6, 2, 1,  # right
            5, 4, 7, 7, 6, 5,  # back
            4, 0, 3, 3, 7, 4,  # left
            3, 2, 6, 6, 7, 3,  # top
            4, 5, 1, 1, 0, 4,  # bottom
        ], dtype='i4')

        # Simplified vertex shader that only uses position and color
        vertex_shader = '''
            #version 330
            
            in vec3 in_position;
            in vec3 in_color;
            
            out vec3 v_color;
            
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            
            void main() {
                gl_Position = projection * view * model * vec4(in_position, 1.0);
                v_color = in_color;
            }
        '''

        fragment_shader = '''
            #version 330
            
            in vec3 v_color;
            out vec4 fragColor;
            
            void main() {
                fragColor = vec4(v_color, 1.0);
            }
        '''

        try:
            self.prog = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader,
            )
            print("Shader program compiled successfully")
        except Exception as e:
            print(f"Shader compilation error: {e}")
            raise

        # Create debug cube buffers
        self.debug_vbo = self.ctx.buffer(self.debug_vertices.tobytes())
        self.debug_ibo = self.ctx.buffer(self.debug_indices.tobytes())
        
        try:
            self.debug_vao = self.ctx.vertex_array(
                self.prog,
                [
                    (self.debug_vbo, '3f 3f', 'in_position', 'in_color'),
                ],
                self.debug_ibo
            )
            print("Debug VAO created successfully")
        except Exception as e:
            print(f"VAO creation error: {e}")
            raise

        # Prepare model data
        self.vertices = mesh.vertices.astype('f4')
        # Create colors for the model (using normals as colors for debugging)
        self.colors = (mesh.vertex_normals * 0.5 + 0.5).astype('f4')
        
        # Combine vertices and colors
        self.model_data = np.hstack([self.vertices, self.colors])
        self.faces = mesh.faces.astype('i4')

        print(f"Model stats:")
        print(f"Vertices: {len(self.vertices)}")
        print(f"Faces: {len(self.faces)}")
        print(f"First few vertices: {self.vertices[:3]}")

        try:
            self.model_vbo = self.ctx.buffer(self.model_data.tobytes())
            self.model_ibo = self.ctx.buffer(self.faces.tobytes())
            
            self.model_vao = self.ctx.vertex_array(
                self.prog,
                [
                    (self.model_vbo, '3f 3f', 'in_position', 'in_color'),
                ],
                self.model_ibo
            )
            print("Model VAO created successfully")
        except Exception as e:
            print(f"Model buffer creation error: {e}")
            raise

        # Enable depth testing
        self.ctx.enable(moderngl.DEPTH_TEST)

        # Camera views
        self.views = {
            '1': {'pos': [0, 0, 3], 'name': 'Front'},
            '2': {'pos': [3, 0, 0], 'name': 'Right'},
            '3': {'pos': [0, 3, 0], 'name': 'Top'},
            '4': {'pos': [2, 2, 2], 'name': 'Isometric'},
        }

    # ... (keep the look_at and perspective methods the same)

    def render_frame(self, camera_pos):
        try:
            # Set up matrices
            view = self.look_at(camera_pos)
            projection = self.perspective(45, self.width/self.height, 0.1, 100.0)
            model = np.identity(4, dtype=np.float32)

            # Update uniforms
            self.prog['model'].write(model.tobytes())
            self.prog['view'].write(view.tobytes())
            self.prog['projection'].write(projection.tobytes())

            # Clear
            self.ctx.clear(0.2, 0.2, 0.2)

            # Draw debug cube
            self.debug_vao.render()

            # Draw model
            self.model_vao.render()

            return True
        except Exception as e:
            print(f"Render error: {e}")
            return False

    def look_at(self, eye, target=[0, 0, 0], up=[0, 1, 0]):
        forward = np.array(target) - np.array(eye)
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        view_matrix = np.identity(4, dtype=np.float32)
        view_matrix[:3, 0] = right
        view_matrix[:3, 1] = up
        view_matrix[:3, 2] = -forward
        view_matrix[:3, 3] = -np.array([
            np.dot(right, eye),
            np.dot(up, eye),
            np.dot(-forward, eye)
        ])
        
        return view_matrix

    def perspective(self, fovy, aspect, near, far):
        fovy = np.radians(fovy)
        f = 1.0 / np.tan(fovy / 2)
        
        projection = np.zeros((4, 4), dtype=np.float32)
        projection[0, 0] = f / aspect
        projection[1, 1] = f
        projection[2, 2] = (far + near) / (near - far)
        projection[2, 3] = 2 * far * near / (near - far)
        projection[3, 2] = -1
        
        return projection

    def render_interactive(self):
        running = True
        current_view = '1'
        
        print("\nControls:")
        for key, view in self.views.items():
            print(f"Press {key} for {view['name']} view")
        print("Press ESC to quit")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    for key in self.views.keys():
                        if event.unicode == key:
                            current_view = key
                            print(f"Switched to {self.views[key]['name']} view")

            camera_pos = self.views[current_view]['pos']
            if not self.render_frame(camera_pos):
                print("Frame render failed")
                break

            pygame.display.flip()

        pygame.quit()


class CubeRenderer:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)

        self.ctx = moderngl.create_context()
        print(f"OpenGL Version: {self.ctx.version_code}")

        # Debug settings
        self.debug_mode = True
        self.show_axes = True
        self.wireframe_mode = False

        # Create a proper cube
        size = 1.0
        vertices = []
        colors = []
        
        # Front face (red)
        vertices.extend([
            [-size, -size,  size],
            [ size, -size,  size],
            [ size,  size,  size],
            [-size,  size,  size],
        ])
        colors.extend([[1.0, 0.0, 0.0] for _ in range(4)])
        
        # Back face (green)
        vertices.extend([
            [-size, -size, -size],
            [ size, -size, -size],
            [ size,  size, -size],
            [-size,  size, -size],
        ])
        colors.extend([[0.0, 1.0, 0.0] for _ in range(4)])
        
        # Right face (blue)
        vertices.extend([
            [ size, -size, -size],
            [ size, -size,  size],
            [ size,  size,  size],
            [ size,  size, -size],
        ])
        colors.extend([[0.0, 0.0, 1.0] for _ in range(4)])
        
        # Left face (yellow)
        vertices.extend([
            [-size, -size, -size],
            [-size, -size,  size],
            [-size,  size,  size],
            [-size,  size, -size],
        ])
        colors.extend([[1.0, 1.0, 0.0] for _ in range(4)])
        
        # Top face (cyan)
        vertices.extend([
            [-size,  size, -size],
            [-size,  size,  size],
            [ size,  size,  size],
            [ size,  size, -size],
        ])
        colors.extend([[0.0, 1.0, 1.0] for _ in range(4)])
        
        # Bottom face (magenta)
        vertices.extend([
            [-size, -size, -size],
            [-size, -size,  size],
            [ size, -size,  size],
            [ size, -size, -size],
        ])
        colors.extend([[1.0, 0.0, 1.0] for _ in range(4)])

        # Convert to numpy array and flatten
        vertices = np.array(vertices, dtype='f4').reshape(-1)
        colors = np.array(colors, dtype='f4').reshape(-1)
        
        # Combine vertices and colors
        self.vertices = np.zeros(len(vertices) + len(colors), dtype='f4')
        self.vertices[0::6] = vertices[0::3]  # x positions
        self.vertices[1::6] = vertices[1::3]  # y positions
        self.vertices[2::6] = vertices[2::3]  # z positions
        self.vertices[3::6] = colors[0::3]    # red
        self.vertices[4::6] = colors[1::3]    # green
        self.vertices[5::6] = colors[2::3]    # blue

        # Create indices for each face (6 faces, 2 triangles each)
        self.indices = np.array([
            # Front face (vertices 0,1,2,3)
            0, 1, 2,  2, 3, 0,
            # Right face (vertices 1,5,6,2)
            1, 5, 6,  6, 2, 1,
            # Back face (vertices 5,4,7,6)
            5, 4, 7,  7, 6, 5,
            # Left face (vertices 4,0,3,7)
            4, 0, 3,  3, 7, 4,
            # Top face (vertices 3,2,6,7)
            3, 2, 6,  6, 7, 3,
            # Bottom face (vertices 4,5,1,0)
            4, 5, 1,  1, 0, 4
        ], dtype='i4')

        # Create wireframe indices
        self.wireframe_indices = np.array([
            # Front face
            0, 1, 1, 2, 2, 3, 3, 0,
            # Back face
            4, 5, 5, 6, 6, 7, 7, 4,
            # Connecting edges
            0, 4, 1, 5, 2, 6, 3, 7
        ], dtype='i4')

        # Create axes vertices
        axis_length = 2.0  # Increased length
        self.axes_vertices = np.array([
            # X axis (red)
            0.0, 0.0, 0.0,  1.0, 0.0, 0.0,
            axis_length, 0.0, 0.0,  1.0, 0.0, 0.0,
            # Y axis (green)
            0.0, 0.0, 0.0,  0.0, 1.0, 0.0,
            0.0, axis_length, 0.0,  0.0, 1.0, 0.0,
            # Z axis (blue)
            0.0, 0.0, 0.0,  0.0, 0.0, 1.0,
            0.0, 0.0, axis_length,  0.0, 0.0, 1.0,
        ], dtype='f4')

        vertex_shader = '''
            #version 330
            
            in vec3 in_position;
            in vec3 in_color;
            
            out vec3 v_color;
            out vec3 v_position;
            
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            uniform bool u_is_wireframe;
            
            void main() {
                v_position = (model * vec4(in_position, 1.0)).xyz;
                v_color = u_is_wireframe ? vec3(1.0) : in_color;
                gl_Position = projection * view * model * vec4(in_position, 1.0);
            }
        '''

        fragment_shader = '''
            #version 330
            
            in vec3 v_color;
            in vec3 v_position;
            out vec4 fragColor;
            
            void main() {
                // Enhance the color intensity
                vec3 color = v_color * 1.5;
                fragColor = vec4(color, 1.0);
            }
        '''

        # Create program
        self.prog = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
        )

        # Create buffers
        self.vbo = self.ctx.buffer(self.vertices.tobytes())
        self.ibo = self.ctx.buffer(self.indices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo, '3f 3f', 'in_position', 'in_color'),
            ],
            self.ibo
        )

        # Create axes buffer
        self.axes_vbo = self.ctx.buffer(self.axes_vertices.tobytes())
        self.axes_vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.axes_vbo, '3f 3f', 'in_position', 'in_color'),
            ]
        )

        # Camera settings
        self.camera_distance = 5.0  # Start further back
        self.camera_rotation = [0.0, 0.0]
        self.rotation_speed = 0.0005
        self.zoom_speed = 0.005
        self.last_position = None
        self.target = np.array([0.0, 0.0, 0.0]) 

        # Enable depth testing
        self.ctx.enable(moderngl.DEPTH_TEST)

    def render(self):
        running = True
        
        print("\nControls:")
        print("Arrow keys to rotate camera (hold SHIFT for slower rotation)")
        print("Z/X to zoom in/out (hold SHIFT for slower zoom)")
        print("R to reset camera to front view")
        print("ESC to quit")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        # Reset to front view
                        self.camera_rotation = [0.0, 0.0]
                        self.camera_distance = 5.0

            # Camera control with shift modifier
            keys = pygame.key.get_pressed()
            shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            speed_mod = 0.2 if shift_held else 1.0

            if keys[pygame.K_LEFT]:
                self.camera_rotation[0] -= self.rotation_speed * speed_mod
            if keys[pygame.K_RIGHT]:
                self.camera_rotation[0] += self.rotation_speed * speed_mod
            if keys[pygame.K_UP]:
                self.camera_rotation[1] = min(self.camera_rotation[1] + 
                                           self.rotation_speed * speed_mod, 
                                           np.pi/2 - 0.1)
            if keys[pygame.K_DOWN]:
                self.camera_rotation[1] = max(self.camera_rotation[1] - 
                                           self.rotation_speed * speed_mod, 
                                           -np.pi/2 + 0.1)
            if keys[pygame.K_z]:
                self.camera_distance = max(2.0, self.camera_distance - 
                                        self.zoom_speed * speed_mod)
            if keys[pygame.K_x]:
                self.camera_distance = min(10.0, self.camera_distance + 
                                        self.zoom_speed * speed_mod)

            # Get camera position
            camera_pos = self.get_camera_position()
            
            # Set up matrices
            model_matrix = np.identity(4, dtype='f4')
            view_matrix = self.look_at(camera_pos, self.target)
            projection_matrix = self.perspective(45, self.width/self.height, 1.0, 100.0)

            # Update uniforms
            self.prog['model'].write(model_matrix.tobytes())
            self.prog['view'].write(view_matrix.tobytes())
            self.prog['projection'].write(projection_matrix.tobytes())

            # Clear
            self.ctx.clear(0.1, 0.1, 0.1)

            # Draw cube
            self.vao.render(moderngl.TRIANGLES)

            # Draw wireframe if enabled
            if self.debug_mode:
                self.prog['u_is_wireframe'].value = True
                self.ctx.line_width = 2.0
                vao_wireframe = self.ctx.vertex_array(
                    self.prog,
                    [
                        (self.vbo, '3f 3f', 'in_position', 'in_color'),
                    ],
                    self.ctx.buffer(self.wireframe_indices.tobytes())
                )
                vao_wireframe.render(moderngl.LINES)

            pygame.display.flip()

            # Print camera info
            print(f"\rCamera: pos={camera_pos}, "
                  f"rot=[{np.degrees(self.camera_rotation[0]):.1f}°, "
                  f"{np.degrees(self.camera_rotation[1]):.1f}°]", 
                  end="")

        pygame.quit()
        

    def get_camera_position(self):
        """Calculate camera position using spherical coordinates"""
        theta = self.camera_rotation[0]  # horizontal angle (azimuth)
        phi = self.camera_rotation[1]    # vertical angle (elevation)
        
        # Convert spherical to Cartesian coordinates
        x = self.camera_distance * np.cos(phi) * np.sin(theta)
        y = self.camera_distance * np.sin(phi)
        z = self.camera_distance * np.cos(phi) * np.cos(theta)
        
        # Start position is looking at front face (positive z)
        return np.array([x, y, z + self.camera_distance], dtype='f4')

    def look_at(self, eye, target=[0, 0, 0], up=[0, 1, 0]):
        """Create look-at matrix"""
        eye = np.array(eye)
        target = np.array(target)
        up = np.array(up)

        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        view_matrix = np.identity(4, dtype='f4')
        view_matrix[:3, 0] = right
        view_matrix[:3, 1] = up
        view_matrix[:3, 2] = -forward
        view_matrix[:3, 3] = -np.array([
            np.dot(right, eye),
            np.dot(up, eye),
            np.dot(-forward, eye)
        ])
        
        return view_matrix

    def perspective(self, fovy, aspect, near, far):
        # Modified perspective calculation
        fovy = np.radians(fovy)
        f = 1.0 / np.tan(fovy / 2)
        
        perspective_matrix = np.zeros((4, 4), dtype='f4')
        perspective_matrix[0, 0] = f / aspect
        perspective_matrix[1, 1] = f
        perspective_matrix[2, 2] = (far + near) / (near - far)
        perspective_matrix[2, 3] = 2.0 * far * near / (near - far)
        perspective_matrix[3, 2] = -1.0
        
        return perspective_matrix

    
    def draw_wireframe(self):
        # Draw white wireframe
        self.ctx.line_width = 1.0  # Thinner lines
        vao_wireframe = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo, '3f 3f', 'in_position', 'in_color'),
            ],
            self.ctx.buffer(self.wireframe_indices.tobytes())
        )
        # Override colors for wireframe
        self.prog['override_color'] = (1.0, 1.0, 1.0, 1.0)  # White color
        vao_wireframe.render(moderngl.LINES)

if __name__ == "__main__":
    model_path = "/home/link/Downloads/sample.glb"
    # try:
    #     renderer = GLBRenderer(model_path)
    #     renderer.render_interactive()
    # except Exception as e:
    #     print(f"Error occurred: {str(e)}")
    #     import traceback
    #     traceback.print_exc()

    try:
        renderer = CubeRenderer()
        renderer.render()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
