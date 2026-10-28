"""Regenerate the MuJoCo meshes from the original USD assets (requires usd-core)."""

from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source/isaaclab_marl/assets"
DESTINATION = ROOT / "source/mjlab_marl/mjlab_marl/assets/mjcf"


def export_mesh(stage, prim_path, filename, face_ids=None):
    prim = stage.GetPrimAtPath(prim_path)
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    points = np.asarray([transform.Transform(point) for point in points])
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    faces = []
    offset = 0
    for count in counts:
        face = list(indices[offset : offset + count])
        faces.append(face)
        offset += count
    if face_ids is not None:
        faces = [faces[i] for i in face_ids]
    used, inverse = np.unique(np.concatenate(faces), return_inverse=True)
    with (DESTINATION / filename).open("w") as output:
        output.write("# Converted from MARLadona USD assets; Apache-2.0.\n")
        for point in points[used]:
            output.write("v " + " ".join(f"{v:.9g}" for v in point) + "\n")
        offset = 0
        for face in faces:
            count = len(face)
            output.write("f " + " ".join(str(i + 1) for i in inverse[offset : offset + count]) + "\n")
            offset += count


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    agent = Usd.Stage.Open(str(SOURCE / "soccer_agent/soccer_agent.usd"))
    export_mesh(agent, "/soccer_agent/base/visuals/soccer_agent_circle/mesh", "agent.obj")
    export_mesh(agent, "/soccer_agent/base/visuals/texture/mesh", "heading.obj")
    ball = Usd.Stage.Open(str(SOURCE / "ball/ball.usd"))
    mesh_path = "/ball/ball/visuals/ball/mesh"
    for subset, color in (("Bianco", "white"), ("Nero_001", "black")):
        face_ids = UsdGeom.Subset(ball.GetPrimAtPath(f"{mesh_path}/{subset}")).GetIndicesAttr().Get()
        export_mesh(ball, mesh_path, f"ball_{color}.obj", face_ids)


if __name__ == "__main__":
    main()
