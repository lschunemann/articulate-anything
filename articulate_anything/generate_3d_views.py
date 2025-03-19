import numpy as np
import trimesh
import moderngl
import cv2
import os
import PIL.Image
import moderngl_window as mglw

class HeadlessWindow(mglw.WindowConfig):
    gl_version = (3, 3)
    window_size = (640, 480)
    resource_dir = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def load_mesh(mesh_path):
    """Load mesh using trimesh"""
    scene = trimesh.load(mesh_path)
    if isinstance(scene, trimesh.Scene):
        mesh = scene.geometry['geometry_0']
        return mesh
    return scene

def create_camera_poses():
    """Create 4 camera poses around the object"""
    poses = []
    distance = 2.0
    angles = [0, 90, 180, 270]  # degrees
    
    for angle in angles:
        theta = np.radians(angle)
        cam_pos = np.array([
            distance * np.sin(theta),
            0,
            distance * np.cos(theta)
        ])
        
        pose = np.eye(4)
        pose[:3, 3] = cam_pos
        
        z = -cam_pos / np.linalg.norm(cam_pos)
        x = np.cross(np.array([0, 1, 0]), z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        
        pose[:3, 0] = x
        pose[:3, 1] = y
        pose[:3, 2] = z
        
        poses.append(pose)
    
    return poses

def prepare_mesh(mesh):
    """Center and scale mesh to fit in view"""
    # Get mesh bounds
    bounds = mesh.bounds
    
    # Calculate center and scale
    center = (bounds[0] + bounds[1]) / 2
    scale = 2.0 / np.max(bounds[1] - bounds[0])  # Scale up to make object more visible
    
    # Center and scale vertices
    vertices = mesh.vertices - center
    vertices = vertices * scale
    
    print(f"Original bounds: {bounds}")
    print(f"Center: {center}")
    print(f"Scale factor: {scale}")
    print(f"New bounds: [{np.min(vertices, axis=0)}, {np.max(vertices, axis=0)}]")
    
    return vertices, mesh.faces

def create_camera_poses():
    """Create 4 camera poses around the object"""
    poses = []
    distance = 3.0  # Increased distance
    height = 0.0    # Set to 0 for debugging
    angles = [0, 90, 180, 270]
    
    for angle in angles:
        theta = np.radians(angle)
        cam_pos = np.array([
            distance * np.sin(theta),
            height,
            distance * np.cos(theta)
        ])
        
        # Look at center point
        target = np.zeros(3)
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(np.array([0, 1, 0]), forward)
        right = right / np.linalg.norm(right)
        
        up = np.cross(forward, right)
        
        # Create view matrix
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = forward
        pose[:3, 3] = cam_pos
        
        print(f"\nCamera {angles.index(angle)} position: {cam_pos}")
        print(f"Looking direction: {forward}")
        
        poses.append(pose)
    
    return poses

def render_views(mesh, output_dir):
    """Render multiple views of the mesh"""
    os.makedirs(output_dir, exist_ok=True)
    
    width, height = 320, 240
    
    # Initialize ModernGL context
    ctx = moderngl.create_standalone_context()
    
    # Create framebuffer with matching dimensions for both color and depth
    color_texture = ctx.texture((width, height), 4)
    depth_texture = ctx.depth_texture((width, height))  # Make sure dimensions match
    fbo = ctx.framebuffer(
        color_attachments=[color_texture],
        depth_attachment=depth_texture
    )
    fbo.viewport = (0, 0, width, height)
    
    # Get mesh vertices and faces
    vertices, faces = prepare_mesh(mesh)
    
    # Create buffers
    vertex_data = vertices.astype('f4').tobytes()
    vbo = ctx.buffer(vertex_data)
    index_data = faces.astype('i4').tobytes()
    ibo = ctx.buffer(index_data)
    
    # Updated shader program with depth output
    prog = ctx.program(
        vertex_shader='''
            #version 330
            
            in vec3 in_position;
            uniform mat4 mvp;
            
            void main() {
                vec4 pos = mvp * vec4(in_position, 1.0);
                gl_Position = pos;
            }
        ''',
        fragment_shader='''
            #version 330
            
            out vec4 fragColor;
            
            void main() {
                fragColor = vec4(1.0, 1.0, 1.0, 1.0);
            }
        '''
    )
    
    vao = ctx.vertex_array(prog, [(vbo, '3f', 'in_position')], ibo)
    
    aspect_ratio = width / height
    fov = 60.0
    near = 0.1
    far = 6.0  # Reduced far plane to match camera distance
    
    f = 1.0 / np.tan(np.radians(fov) / 2)
    proj = np.array([
        [f/aspect_ratio, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near)/(near - far), (2*far*near)/(near - far)],
        [0.0, 0.0, -1.0, 0.0]
    ], dtype='f4')
    
    # Updated shader program with proper color output
    prog = ctx.program(
        vertex_shader='''
            #version 330
            
            in vec3 in_position;
            uniform mat4 mvp;
            uniform mat4 model;
            
            out vec3 v_position;
            
            void main() {
                v_position = (model * vec4(in_position, 1.0)).xyz;
                gl_Position = mvp * vec4(in_position, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330
            
            in vec3 v_position;
            
            out vec4 fragColor;
            
            void main() {
                // Simple shading based on position
                vec3 color = normalize(v_position) * 0.5 + 0.5;
                fragColor = vec4(color, 1.0);
            }
        '''
    )

    poses = create_camera_poses()
    
    for idx, pose in enumerate(poses):
        # Clear buffers
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        fbo.use()
        
        # Calculate matrices
        view = np.linalg.inv(pose)
        model = np.eye(4, dtype='f4')
        mvp = proj @ view @ model
        
        # Set uniforms
        prog['mvp'].write(mvp.astype('f4').tobytes())
        prog['model'].write(model.astype('f4').tobytes())
        
        # Render
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        vao.render(moderngl.TRIANGLES)
        
        # Read buffers
        color_buffer = fbo.read(components=4)
        depth_buffer = fbo.read(attachment=-1)
        
        # Reshape color buffer
        color = np.frombuffer(color_buffer, dtype='u1').reshape(height, width, 4)
        
        # Process depth buffer with safe calculations
        depth = np.frombuffer(depth_buffer, dtype='f4')
        depth_width = int(np.sqrt(len(depth)))
        depth = depth.reshape(depth_width, depth_width)
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Convert depth values safely
        z_ndc = np.clip(depth, 0, 1) * 2.0 - 1.0
        z_eye = np.zeros_like(z_ndc)
        valid_mask = (far + near - z_ndc * (far - near)) != 0
        z_eye[valid_mask] = 2.0 * near * far / (far + near - z_ndc[valid_mask] * (far - near))
        
        # Normalize depth values safely
        depth_min = np.nanmin(z_eye)
        depth_max = np.nanmax(z_eye)
        
        if depth_max > depth_min and not np.isnan(depth_min) and not np.isnan(depth_max):
            depth_normalized = (z_eye - depth_min) / (depth_max - depth_min)
            depth_scaled = (depth_normalized * 65535).astype(np.uint16)
        else:
            depth_scaled = np.zeros_like(z_eye, dtype=np.uint16)
        
        # Save outputs
        color_path = os.path.join(output_dir, f'view_{idx}_rgb.png')
        depth_path = os.path.join(output_dir, f'view_{idx}_depth.png')
        
        PIL.Image.fromarray(color).save(color_path)
        cv2.imwrite(depth_path, depth_scaled)
        
        # Debug output
        print(f"\nView {idx} stats:")
        print(f"Color shape: {color.shape}")
        print(f"Depth shape: {depth.shape}")
        print(f"Color range: {color.min()} to {color.max()}")
        print(f"Raw depth range: {depth.min():.3f} to {depth.max():.3f}")
        print(f"Z-eye range: {np.nanmin(z_eye):.3f} to {np.nanmax(z_eye):.3f}")
    
    # Clean up
    vbo.release()
    ibo.release()
    prog.release()
    vao.release()
    fbo.release()
    ctx.release()

def main(mesh_path, output_dir):
    """Main function to process mesh and generate views"""
    try:
        # Load and print mesh info
        mesh = load_mesh(mesh_path)
        print("\nMesh Information:")
        print(f"Vertices: {len(mesh.vertices)}")
        print(f"Faces: {len(mesh.faces)}")
        print(f"Bounds: {mesh.bounds}")
        
        # Render views
        render_views(mesh, output_dir)
        
        print(f"\nSuccessfully generated views in {output_dir}")
    except Exception as e:
        print(f"Error processing mesh: {str(e)}")
        raise e

if __name__ == "__main__":
    mesh_path = "/home/link/Downloads/sample.glb"
    output_dir = "../datasets/output_views"
    main(mesh_path, output_dir)