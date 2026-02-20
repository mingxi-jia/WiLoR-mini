import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import trimesh
import pyrender
import time

from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

# --- Lighting Helpers (Simplified for Speed) ---
def create_raymond_lights():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes = []
    for phi, theta in zip(phis, thetas):
        matrix = np.eye(4)
        # Simplified rotation logic for speed
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))
    return nodes

class Renderer:
    def __init__(self, faces: np.array, width=640, height=480):
        # Additional faces from your original script
        faces_new = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239],
                              [239, 122, 279], [122, 118, 279], [279, 118, 215],
                              [118, 117, 215], [215, 117, 214], [117, 119, 214],
                              [214, 119, 121], [119, 120, 121], [121, 120, 78],
                              [120, 108, 78], [78, 108, 79]])
        self.faces = np.concatenate([faces, faces_new], axis=0)
        self.faces_left = self.faces[:, [0, 2, 1]]
        
        # CRITICAL: Initialize the renderer ONCE here
        self.width = width
        self.height = height
        self.renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        
        # Pre-create lighting nodes to avoid reallocation
        self.light_nodes = create_raymond_lights()

    def render_rgba(self, vertices, cam_t, is_right, focal_length, mesh_base_color=(0.6, 0.2, 0.6)):
        # Convert vertices to mesh
        # We assume vertices are already in the camera coordinate system or relative
        # Note: WiLoR pred_vertices are usually in weak-perspective or cropped space
        
        # Flip color if right hand (as per your logic)
        color = mesh_base_color[::-1] if is_right else mesh_base_color
        v_colors = np.array([(*color, 1.0)] * vertices.shape[0])
        
        faces = self.faces if is_right else self.faces_left
        t_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=v_colors)
        
        # Fix orientation (OpenCV vs OpenGL Y-axis)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        t_mesh.apply_transform(rot)
        
        mesh = pyrender.Mesh.from_trimesh(t_mesh)
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh)

        # Camera setup
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = cam_t
        camera_pose[0, 3] *= -1 # Match your original flip
        
        camera = pyrender.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=self.width / 2., cy=self.height / 2., zfar=1e5
        )
        scene.add(camera, pose=camera_pose)

        # Add pre-defined lights
        for node in self.light_nodes:
            scene.add_node(node)

        # Render - this is now much faster because context is already open
        color_img, _ = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        return color_img.astype(np.float32) / 255.0

# --- Main Execution ---

device = torch.device("cuda")
# WiLoR performs best at float16 on modern NVIDIA GPUs
dtype = torch.float16 

pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
renderer = Renderer(pipe.wilor_model.mano.faces, width=640, height=480)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

prev_time = time.time()

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue

        frame = np.asanyarray(color_frame.get_data())
        
        # 1. Faster Inference with No Grad & Autocast
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = pipe.predict(frame)

        # 2. Efficient Compositing
        render_image = frame.copy().astype(np.float32) / 255.0

        for out in outputs:
            # Extract WiLoR specific outputs
            preds = out["wilor_preds"]
            verts = preds["pred_vertices"][0]
            cam_t = preds["pred_cam_t_full"][0]
            focal = preds["scaled_focal_length"]
            
            # Render the overlay
            rgba_mask = renderer.render_rgba(verts, cam_t, out["is_right"], focal)
            
            # Alpha blending (Vectorized)
            alpha = rgba_mask[:, :, 3:]
            render_image = (render_image * (1 - alpha)) + (rgba_mask[:, :, :3] * alpha)

        # 3. FPS Calculation and Display
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        display_frame = (render_image * 255).astype(np.uint8)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("WiLoR Optimized", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()