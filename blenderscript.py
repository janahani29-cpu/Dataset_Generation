import bpy
import time
import math
import random
from mathutils import Euler
from pathlib import Path
from bpy_extras.object_utils import world_to_camera_view

def randomly_rotate_object(obj):
    """Apply random rotation to the object."""
    obj.rotation_euler = Euler((0,0,random.uniform(0, math.pi * 2)), 'XYZ')

def randomly_move_object(obj, x_range=(-1, 1), y_range=(-1, 1), z_range=(0, 0)):
    """Move the object to a random XYZ location within ranges."""
    obj.location.x = random.uniform(*x_range)
    obj.location.y = random.uniform(*y_range)
    obj.location.z = random.uniform(*z_range)


# YOLO LABEL 

def generate_yolo_label(obj, camera):
    """
    Returns YOLO-format bounding box for object:
    class x_center y_center width height  (all normalized 0–1)
    """
    scene = bpy.context.scene
    mesh = obj.data

    coords_2d = []

    # Convert all vertices to camera-view normalized coordinates
    for v in mesh.vertices:
        world_coord = obj.matrix_world @ v.co
        co_2d = world_to_camera_view(scene, camera, world_coord)

        
        coords_2d.append(co_2d)

    xs = [c.x for c in coords_2d]
    ys = [c.y for c in coords_2d]

    # bounding box
    min_x = max(min(xs), 0.0)
    max_x = min(max(xs), 1.0)
    min_y = max(min(ys), 0.0)
    max_y = min(max(ys), 1.0)

    if min_x >= max_x or min_y >= max_y:
        return None  # out of view

    # YOLO format
    x_center = (min_x + max_x) / 2
    y_center = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y

    return (0, x_center, y_center, width, height) 



obj_name = "ANKH"
obj = bpy.context.scene.objects[obj_name]

splits = [('train', 50),('val',   10),('test',  5)]

base_output = Path(r"E:\eagles\blender\project_2")
for split, _ in splits:
    (base_output / split / obj_name).mkdir(parents=True, exist_ok=True)

camera = bpy.context.scene.camera
scene = bpy.context.scene

img_w = scene.render.resolution_x
img_h = scene.render.resolution_y

# RENDER LOOP

start_idx = 0
start_time = time.time()

for split_name, render_count in splits:

    print(f"\n=== Starting split: {split_name} ({render_count} renders) ===")

    obj.hide_render = False
    split_folder = base_output / split_name / obj_name

    for i in range(render_count):
        print(f"\nRendering {split_name} image {i+1}/{render_count}")

        # Randomize object pose
        randomly_rotate_object(obj)
        randomly_move_object(obj,
            x_range=(-14, 14),
            y_range=(-8, 8),
            z_range=(0.163299, 0.163299)
        )

        # Estimate time
        elapsed = time.time() - start_time
        avg_time = elapsed / (start_idx + 1)
        remaining = avg_time * (render_count - i - 1)
        print(f"Estimated time left: {time.strftime('%H:%M:%S', time.gmtime(remaining))}")

        # File paths
        img_name = f"{i:03d}.png"
        label_name = f"{i:03d}.txt"

        output_image_path = (split_folder / img_name).as_posix()
        output_label_path = split_folder / label_name

        # Render image
        bpy.context.scene.render.filepath = output_image_path
        bpy.ops.render.render(write_still=True)

        # YOLO label creation
        yolo_bbox = generate_yolo_label(obj, camera)

        with open(output_label_path, "w") as f:
            if yolo_bbox is not None:
                cls, xc, yc, w, h = yolo_bbox
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            else:
                print("WARNING: Object not visible → empty label")
        start_idx += 1

    obj.hide_render = True

print("\nDONE")
print(f"Total Rendered Images: {start_idx}")
print(f"Total Time: {time.time() - start_time:.2f} seconds")
