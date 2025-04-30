import open3d as o3d
import numpy as np
from argparse import ArgumentParser
import pickle
import trimesh
import cv2
import json
import torch
import os


from PIL import Image

from utils.align_util import (
    render_multi_images,
    render_image,
    as_mesh,
    project_2d_to_3d,
    plot_mesh_with_points,
    plot_image_with_points,
    select_point,
)
from match_pairs import image_pair_matching
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import KDTree

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def pose_selection_render_superglue(raw_img, fov, mesh_path, mesh, crop_img, output_dir, device="cpu"):
    # Calculate suitable rendering radius
    bounding_box = mesh.bounds
    max_dimension = np.linalg.norm(bounding_box[1] - bounding_box[0])
    radius = 2 * (max_dimension / 2) / np.tan(fov / 2)

    # Render multimle images and feature matching
    colors, depths, camera_poses, camera_intrinsics = render_multi_images(
        mesh_path,
        raw_img.shape[1],
        raw_img.shape[0],
        fov,
        radius=radius,
        num_samples=8,
        num_ups=4,
        device=device,
    )
    grays = [cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) for color in colors]
    # Use superglue to match the features
    best_idx, match_result = image_pair_matching(
        grays, crop_img, output_dir, viz_best=True
    )
    print("matched point number", np.sum(match_result["matches"] > -1))

    best_color = colors[best_idx]
    best_depth = depths[best_idx]
    best_pose = camera_poses[best_idx].cpu().numpy()
    return best_color, best_depth, best_pose, match_result, camera_intrinsics


def registration_pnp(mesh_matching_points, raw_matching_points, intrinsic):
    # Solve the PNP and verify the reprojection error
    success, rvec, tvec = cv2.solvePnP(
        np.float32(mesh_matching_points),
        np.float32(raw_matching_points),
        np.float32(intrinsic),
        distCoeffs=np.zeros(4, dtype=np.float32),
        flags=cv2.SOLVEPNP_EPNP,
    )
    assert success, "solvePnP failed"
    projected_points, _ = cv2.projectPoints(
        np.float32(mesh_matching_points),
        rvec,
        tvec,
        intrinsic,
        np.zeros(4, dtype=np.float32),
    )
    error = np.linalg.norm(
        np.float32(raw_matching_points) - projected_points.reshape(-1, 2), axis=1
    ).mean()
    print(f"Reprojection Error: {error}")
    if error > 50:
        print(f"solvePnP failed for this case {case_name}.$$$$$$$$$$$$$$$$$$$$$$$$$$")

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    mesh2raw_camera = np.eye(4, dtype=np.float32)
    mesh2raw_camera[:3, :3] = rotation_matrix
    mesh2raw_camera[:3, 3] = tvec.squeeze()

    return mesh2raw_camera


def registration_scale(mesh_matching_points_cam, matching_points_cam):
    # After PNP, optimize the scale in the camera coordinate
    def objective(scale, mesh_points, pcd_points):
        transformed_points = scale * mesh_points
        loss = np.sum(np.sum((transformed_points - pcd_points) ** 2, axis=1))
        return loss

    initial_scale = 1
    result = minimize(
        objective,
        initial_scale,
        args=(mesh_matching_points_cam, matching_points_cam),
        method="L-BFGS-B",
    )
    optimal_scale = result.x[0]
    print("Rescale:", optimal_scale)
    return optimal_scale


def deform_ARAP(initial_mesh_world, mesh_matching_points_world, matching_points):
    # Do the ARAP deformation based on the matching keypoints
    mesh_vertices = np.asarray(initial_mesh_world.vertices)
    kdtree = KDTree(mesh_vertices)
    _, mesh_points_indices = kdtree.query(mesh_matching_points_world)
    mesh_points_indices = np.asarray(mesh_points_indices, dtype=np.int32)
    deform_mesh = initial_mesh_world.deform_as_rigid_as_possible(
        o3d.utility.IntVector(mesh_points_indices),
        o3d.utility.Vector3dVector(matching_points),
        max_iter=1,
    )
    return deform_mesh, mesh_points_indices


def get_matching_ray_registration(
    mesh_world, obs_points_world, mesh, trimesh_indices, c2w, w2c
):
    # Get the matching indices and targets based on the viewpoint
    obs_points_cam = np.dot(
        w2c,
        np.hstack((obs_points_world, np.ones((obs_points_world.shape[0], 1)))).T,
    ).T
    obs_points_cam = obs_points_cam[:, :3]
    vertices_cam = np.dot(
        w2c,
        np.hstack(
            (
                np.asarray(mesh_world.vertices),
                np.ones((np.asarray(mesh_world.vertices).shape[0], 1)),
            )
        ).T,
    ).T
    vertices_cam = vertices_cam[:, :3]

    obs_kd = KDTree(obs_points_cam)

    new_indices = []
    new_targets = []
    # trimesh used to do the ray-casting test
    mesh.vertices = np.asarray(vertices_cam)[trimesh_indices]
    for index, vertex in enumerate(vertices_cam):
        ray_origins = np.array([[0, 0, 0]])
        ray_direction = vertex
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        ray_directions = np.array([ray_direction])
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
        )

        ignore_flag = False

        if len(locations) > 0:
            first_intersection = locations[0]
            vertex_distance = np.linalg.norm(vertex)
            intersection_distance = np.linalg.norm(first_intersection)
            if intersection_distance < vertex_distance - 1e-4:
                # If the intersection point is not the vertex, it means the vertex is not visible from the camera viewpoint
                ignore_flag = True

        if ignore_flag:
            continue
        else:
            # Select the closest point to the ray of the observation points as the matching point
            indices = obs_kd.query_ball_point(vertex, 0.02)
            line_distances = line_point_distance(vertex, obs_points_cam[indices])
            # Get the closest point
            if len(line_distances) > 0:
                closest_index = np.argmin(line_distances)
                target = np.dot(
                    c2w, np.hstack((obs_points_cam[indices][closest_index], 1))
                )
                new_indices.append(index)
                new_targets.append(target[:3])

    new_indices = np.asarray(new_indices)
    new_targets = np.asarray(new_targets)

    return new_indices, new_targets


def deform_ARAP_ray_registration(
    deform_kp_mesh_world,
    obs_points_world,
    mesh,
    trimesh_indices,
    c2ws,
    w2cs,
    mesh_points_indices,
    matching_points,
):
    final_indices = []
    final_targets = []
    for index, target in zip(mesh_points_indices, matching_points):
        if index not in final_indices:
            final_indices.append(index)
            final_targets.append(target)

    for c2w, w2c in zip(c2ws, w2cs):
        new_indices, new_targets = get_matching_ray_registration(
            deform_kp_mesh_world, obs_points_world, mesh, trimesh_indices, c2w, w2c
        )
        for index, target in zip(new_indices, new_targets):
            if index not in final_indices:
                final_indices.append(index)
                final_targets.append(target)

    # Also need to adjust the positions to make sure they are above the table
    indices = np.where(np.asarray(deform_kp_mesh_world.vertices)[:, 2] > 0)[0]
    for index in indices:
        if index not in final_indices:
            final_indices.append(index)
            target = np.asarray(deform_kp_mesh_world.vertices)[index].copy()
            target[2] = 0
            final_targets.append(target)
        else:
            target = final_targets[final_indices.index(index)]
            if target[2] > 0:
                target[2] = 0
                final_targets[final_indices.index(index)] = target

    final_mesh_world = deform_kp_mesh_world.deform_as_rigid_as_possible(
        o3d.utility.IntVector(final_indices),
        o3d.utility.Vector3dVector(final_targets),
        max_iter=1,
    )
    return final_mesh_world


def line_point_distance(p, points):
    # Compute the distance between points and the line between p and [0, 0, 0]
    p = p / np.linalg.norm(p)
    points_to_origin = points
    cross_product = np.linalg.norm(np.cross(points_to_origin, p), axis=1)
    return cross_product / np.linalg.norm(p)

# Create point clouds by sampling from the mesh surface
def sample_points_from_mesh(mesh, n_points=10000):
    """Sample points uniformly from the mesh surface"""
    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
    
    # Calculate face normals to get colors (optional)
    face_normals = mesh.face_normals[face_indices]
    
    # If the mesh has vertex colors or a texture, you can sample colors too
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        # Sample colors from the mesh if they exist
        vertex_indices = mesh.faces[face_indices]
        colors = mesh.visual.vertex_colors[vertex_indices].mean(axis=1)[:, :3] / 255.0
    else:
        # Otherwise use face normals converted to colors (normalized from -1,1 to 0,1)
        colors = (face_normals + 1) / 2

    return points, colors

def read_masks(object_masks_path, new_shape):
    masks = []

    files = os.listdir(object_masks_path)
    files.sort()
    # for each file in object_masks_path read the binary image and convert to numpy
    for file in files:
        object_mask = np.array(Image.open(os.path.join(object_masks_path, file)))

        object_mask = cv2.resize(object_mask.astype(np.uint8), new_shape, interpolation=cv2.INTER_NEAREST)
        object_mask = object_mask.astype(bool)

        masks.append(object_mask)

    return np.array(masks)

if __name__ == "__main__":


    parser = ArgumentParser()
    # required
    parser.add_argument("--base_path", type=str, default="/home/link/DreMa/third_party/articulate-anything/datasets/output_views")
    parser.add_argument("--case_name", type=str, default="microwave_multi-view")

    # optional
    parser.add_argument("--mesh_name", type=str, default="microwave_multi-view.glb")
    parser.add_argument("--image_name", type=str, default="render_microwave_multi-view_2_0031")
    parser.add_argument("--output_dir", type=str, default="../temp/Video1/matching_temp")
    parser.add_argument("--mask_directory", type=str, default="../../segmentation_masks/microwave_multi-view")
    parser.add_argument("--image_directory", type=str, default="")
    parser.add_argument("--depth_path", type=str, default="")
    parser.add_argument("--pcd_directory", type=str, default="pcd")
    parser.add_argument("--intrinsic_file", type=str, default="camera_params_microwave_multi-view_2.npz")
    parser.add_argument("--extrinsic_file", type=str, default="extrinsics.npz")
    parser.add_argument("--convert_extrinsic", type=bool, default=True)
    parser.add_argument("--vis", type=bool, default=True)
    parser.add_argument("--use_gpu_for_rendering", type=bool, default=False)

    args = parser.parse_args()

    VIS = args.vis

    # set root path
    base_path = args.base_path
    case_name = args.case_name
    root_dir = os.path.join(base_path, case_name)
    print("Root path: ", root_dir)

    # set rgb image path using os
    rgb_path = os.path.join(base_path, case_name, args.image_directory, f"{args.image_name}.png")
    print("RGB path", rgb_path)

    # set mask path using os
    mask_path = os.path.join(base_path, case_name, args.mask_directory, f"{args.image_name}.png")
    print("Mask path", mask_path)

    # # set point cloud path using os
    # pcd_path = os.path.join(base_path, case_name, args.pcd_directory, "full_object_points.pkl")
    # pcd_image_path = os.path.join(base_path, case_name, args.pcd_directory, f"{args.image_name}.pkl")
    # print("PCD path", pcd_path)
    # print("RGB PCD path", pcd_image_path)

    # read the depth image
    depth_path = os.path.join(base_path, case_name, args.depth_path, f"{args.image_name}.png")
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # set mesh path using os
    mesh_path = os.path.join(base_path, case_name, args.mesh_name)
    print("Mesh path", mesh_path)

    # set intrinsic path using os
    intrinsic_path = os.path.join(base_path, case_name, args.intrinsic_file)
    print("Intrinsic path", intrinsic_path)

    # set extrinsic path using os
    extrinsic_path = os.path.join(base_path, case_name, args.extrinsic_file)
    print("Extrinsic path", extrinsic_path)

    # set output dir
    output_dir = args.output_dir
    existDir(output_dir)
    print("Output dir", output_dir)

    # load rgb image, mask
    raw_img = cv2.imread(rgb_path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # # Load the pcd in world coordinate of the object
    # with open(pcd_path, "rb") as f:
    #     data = pickle.load(f)
    #     obs_points = data["points"]
    #     obs_colors = data["colors"]

    # Load the pcd in world coordinate of the object
    # with open(pcd_image_path, "rb") as f:
    #     data = pickle.load(f)
    #     first_points = data["points"]
    #     first_colors = data["colors"]
    #     first_mask = np.ones(len(first_points), dtype=bool)

    # if the mask is between 0 and 1, convert to 0 and 255
    if np.max(mask_img) <= 1:
        mask_img = mask_img * 255

    # Load the intrinsic and extrinsic parameters
    # with open(extrinsic_path, "rb") as f:
    #     extrinsics = pickle.load(f)
    #     c2ws = extrinsics["cam_c2w"]
    #     if args.cover_extrinsic:
    #         w2cs = np.array([np.linalg.inv(c2w) for c2w in c2ws])

    camera_data = np.load(intrinsic_path)
    intrinsic = camera_data['K']
    R = camera_data['R']
    t = camera_data['t']

    # Create the camera-to-world matrix (c2w)
    c2w = np.eye(4)  # Create a 4x4 identity matrix
    c2w[:3, :3] = R  # Add rotation component
    c2w[:3, 3] = t.flatten()  # Add translation component

    # Create the world-to-camera matrix (w2c)
    w2c = np.linalg.inv(c2w)

    # If you have multiple camera views, you need to create c2ws and w2cs arrays
    c2ws = [c2w]  # If you have just one view
    w2cs = [w2c]  # If you have just one view

    # Load the shape prior
    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    mesh = as_mesh(mesh)

    # Sample points from the mesh surface
    n_sample_points = 10000  # Adjust as needed
    obs_points, obs_colors = sample_points_from_mesh(mesh, n_sample_points)

    # Create a first view point cloud (similar to the original first_points)
    # This can be a subset or a transformed version of the original points
    first_points = obs_points.copy()
    first_colors = obs_colors.copy()

    # If you need to transform the points to match your camera view:
    # first_points = np.dot(mesh2world[:3, :3], obs_points.T).T + mesh2world[:3, 3]

    # Create a mask for the first points (all valid in this case)
    first_mask = np.ones(len(first_points), dtype=bool)

    # Calculate camera parameters
    fov = 2 * np.arctan(raw_img.shape[1] / (2 * intrinsic[0, 0]))

    if not os.path.exists(f"{output_dir}/best_match.pkl"):
        # 2D feature Matching to get the best pose of the object
        bbox = np.argwhere(mask_img > 0.8 * 255)
        print(bbox)
        bbox = (
            np.min(bbox[:, 1]),
            np.min(bbox[:, 0]),
            np.max(bbox[:, 1]),
            np.max(bbox[:, 0]),
        )
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = (
            int(center[0] - size // 2),
            int(center[1] - size // 2),
            int(center[0] + size // 2),
            int(center[1] + size // 2),
        )
        # Make sure the bounding box is within the image
        bbox = (
            max(0, bbox[0]),
            max(0, bbox[1]),
            min(raw_img.shape[1], bbox[2]),
            min(raw_img.shape[0], bbox[3]),
        )
        # Get the masked cropped image used for superglue
        crop_img = raw_img.copy()
        mask_bool = mask_img > 0
        crop_img[~mask_bool] = 0
        crop_img = crop_img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)

        # Render the object and match the features
        best_color, best_depth, best_pose, match_result, camera_intrinsics = (
            pose_selection_render_superglue(
                raw_img,
                fov,
                mesh_path,
                mesh,
                crop_img,
                output_dir=output_dir,
                device="cuda" if args.use_gpu_for_rendering else "cpu",
            )
        )
        with open(f"{output_dir}/best_match.pkl", "wb") as f:
            pickle.dump(
                [
                    best_color,
                    best_depth,
                    best_pose,
                    match_result,
                    camera_intrinsics,
                    bbox,
                ],
                f,
            )
    else:
        with open(f"{output_dir}/best_match.pkl", "rb") as f:
            best_color, best_depth, best_pose, match_result, camera_intrinsics, bbox = (
                pickle.load(f)
            )

    # Process to get the matching points on the mesh and on the image
    # Get the projected 3D matching points on the mesh
    valid_matches = match_result["matches"] > -1
    render_matching_points = match_result["keypoints0"][valid_matches]
    mesh_matching_points, valid_mask = project_2d_to_3d(
        render_matching_points, best_depth, camera_intrinsics, best_pose
    )
    render_matching_points = render_matching_points[valid_mask]
    # Get the matching points on the raw image
    raw_matching_points_box = match_result["keypoints1"][
        match_result["matches"][valid_matches]
    ]
    raw_matching_points_box = raw_matching_points_box[valid_mask]
    raw_matching_points = raw_matching_points_box + np.array([bbox[0], bbox[1]])

    if VIS:
        # Do visualization for the matching
        plot_mesh_with_points(
            mesh,
            mesh_matching_points,
            f"{output_dir}/mesh_matching.png",
        )
        plot_image_with_points(
            best_depth,
            render_matching_points,
            f"{output_dir}/render_matching.png",
        )
        plot_image_with_points(
            raw_img,
            raw_matching_points,
            f"{output_dir}/raw_matching.png",
        )

    # Do PnP optimization to optimize the rotation between the 3D mesh keypoints and the 2D image keypoints
    mesh2raw_camera = registration_pnp(
        mesh_matching_points, raw_matching_points, intrinsic
    )

    if VIS:
        pnp_camera_pose = np.eye(4, dtype=np.float32)
        pnp_camera_pose[:3, :3] = np.linalg.inv(mesh2raw_camera[:3, :3])
        pnp_camera_pose[3, :3] = mesh2raw_camera[:3, 3]
        pnp_camera_pose[:, :2] = -pnp_camera_pose[:, :2]
        color, depth = render_image(
            mesh_path, pnp_camera_pose, raw_img.shape[1], raw_img.shape[0], fov, "cuda"
        )
        vis_mask = depth > 0
        color[0][~vis_mask] = raw_img[~vis_mask]
        plt.imsave(f"{output_dir}/pnp_results.png", color[0])

    # Transform the mesh into the real world coordinate
    mesh_matching_points_cam = np.dot(
        mesh2raw_camera,
        np.hstack(
            (mesh_matching_points, np.ones((mesh_matching_points.shape[0], 1)))
        ).T,
    ).T
    mesh_matching_points_cam = mesh_matching_points_cam[:, :3]

    # Find the cloest points for the raw_matching_points
    new_match, matching_points = select_point(
        first_points, raw_matching_points, first_mask
    )
    matching_points_cam = np.dot(
        w2c, np.hstack((matching_points, np.ones((matching_points.shape[0], 1)))).T
    ).T
    matching_points_cam = matching_points_cam[:, :3]

    if VIS:
        # Draw the raw_matching_points and new matching points on the masked
        vis_img = raw_img.copy()
        vis_img[~first_mask] = 0
        plot_image_with_points(
            vis_img,
            raw_matching_points,
            f"{output_dir}/raw_matching_valid.png",
            new_match,
        )

    # Use the matching points in the camera coordinate to optimize the scame between the mesh and the observation
    optimal_scale = registration_scale(mesh_matching_points_cam, matching_points_cam)

    # Compute the rigid transformation from the original mesh to the final world coordinate
    scale_matrix = np.eye(4) * optimal_scale
    scale_matrix[3, 3] = 1
    mesh2world = np.dot(c2w, np.dot(scale_matrix, mesh2raw_camera))

    mesh_matching_points_world = np.dot(
        mesh2world,
        np.hstack(
            (mesh_matching_points, np.ones((mesh_matching_points.shape[0], 1)))
        ).T,
    ).T
    mesh_matching_points_world = mesh_matching_points_world[:, :3]

    # Do the ARAP based on the matching keypoints
    # Convert the mesh to open3d to use the ARAP function
    initial_mesh_world = o3d.geometry.TriangleMesh()
    initial_mesh_world.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    initial_mesh_world.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    # Need to remove the duplicated vertices to enable open3d, however, the duplicated points are important in trimesh for texture
    initial_mesh_world = initial_mesh_world.remove_duplicated_vertices()
    # Get the index from original vertices to the mesh vertices, mapping between trimesh and open3d
    kdtree = KDTree(initial_mesh_world.vertices)
    _, trimesh_indices = kdtree.query(np.asarray(mesh.vertices))
    trimesh_indices = np.asarray(trimesh_indices, dtype=np.int32)
    initial_mesh_world.transform(mesh2world)

    # ARAP based on the keypoints
    deform_kp_mesh_world, mesh_points_indices = deform_ARAP(
        initial_mesh_world, mesh_matching_points_world, matching_points
    )

    # Do the ARAP based on both the ray-casting matching and the keypoints
    # Identify the vertex which blocks or blocked by the observation, then match them with the observation points on the ray
    final_mesh_world = deform_ARAP_ray_registration(
        deform_kp_mesh_world,
        obs_points,
        mesh,
        trimesh_indices,
        c2ws,
        w2cs,
        mesh_points_indices,
        matching_points,
    )

    if VIS:
        final_mesh_world.compute_vertex_normals()

        # Visualize the partial observation and the mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obs_points)
        pcd.colors = o3d.utility.Vector3dVector(obs_colors)

        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Render the final stuffs as a turntable video
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        height, width, _ = dummy_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(
            f"{output_dir}/final_matching.mp4", fourcc, 30, (width, height)
        )
        # final_mesh_world.compute_vertex_normals()
        # final_mesh_world.translate([0, 0, 0.2])
        # mesh_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(final_mesh_world)
        # o3d.visualization.draw_geometries([pcd, final_mesh_world], window_name="Matching")
        vis.add_geometry(pcd)
        vis.add_geometry(final_mesh_world)
        # vis.add_geometry(coordinate)
        view_control = vis.get_view_control()

        for j in range(360):
            view_control.rotate(10, 0)
            vis.poll_events()
            vis.update_renderer()
            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        vis.destroy_window()

    mesh.vertices = np.asarray(final_mesh_world.vertices)[trimesh_indices]
    mesh.export(f"{output_dir}/final_mesh.glb")
