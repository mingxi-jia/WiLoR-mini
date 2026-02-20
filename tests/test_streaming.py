import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import trimesh
import pyrender

from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline


def create_raymond_lights():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))
    return nodes


def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        matrix[:3, 3] = dist * z
        poses.append(matrix)
    return poses


class Renderer:
    def __init__(self, faces: np.array):
        faces_new = np.array([[92, 38, 234],
                              [234, 38, 239],
                              [38, 122, 239],
                              [239, 122, 279],
                              [122, 118, 279],
                              [279, 118, 215],
                              [118, 117, 215],
                              [215, 117, 214],
                              [117, 119, 214],
                              [214, 119, 121],
                              [119, 120, 121],
                              [121, 120, 78],
                              [120, 108, 78],
                              [78, 108, 79]])
        faces = np.concatenate([faces, faces_new], axis=0)
        self.faces = faces
        self.faces_left = self.faces[:, [0, 2, 1]]

    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9),
                            rot_axis=[1, 0, 0], rot_angle=0, is_right=1):
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces_left.copy(), vertex_colors=vertex_colors)
        rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render_rgba(self, vertices, cam_t=None, rot_axis=[1, 0, 0], rot_angle=0,
                    camera_z=3, mesh_base_color=(1.0, 1.0, 0.9), scene_bg_color=(0, 0, 0),
                    render_res=[256, 256], focal_length=None, is_right=None):
        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length / render_res[1]])
        if is_right:
            mesh_base_color = mesh_base_color[::-1]
        mesh = self.vertices_to_trimesh(vertices, np.array([0, 0, 0]), mesh_base_color, rot_axis, rot_angle, is_right=is_right)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self._add_lighting(scene, camera_node)
        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()
        return color

    def _add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)


LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float16

pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
renderer = Renderer(pipe.wilor_model.mano.faces)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())  # BGR
        outputs = pipe.predict(frame)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        render_image = image.copy().astype(np.float32)[:, :, ::-1] / 255.0

        for out in outputs:
            verts = out["wilor_preds"]["pred_vertices"][0]
            is_right = out["is_right"]
            cam_t = out["wilor_preds"]["pred_cam_t_full"][0]
            scaled_focal_length = out["wilor_preds"]["scaled_focal_length"]

            cam_view = renderer.render_rgba(
                verts,
                cam_t=cam_t,
                render_res=[image.shape[1], image.shape[0]],
                is_right=is_right,
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            render_image = (
                render_image[:, :, :3] * (1 - cam_view[:, :, 3:])
                + cam_view[:, :, :3] * cam_view[:, :, 3:]
            )

        render_image = (255 * render_image).astype(np.uint8)
        cv2.imshow("WiLoR RealSense", render_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
