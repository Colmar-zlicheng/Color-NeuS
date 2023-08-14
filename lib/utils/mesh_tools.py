import torch
import trimesh
import numpy as np
import aspose.threed as a3d
from plyfile import PlyData, PlyElement
from pytorch3d.loss import chamfer_distance


def load_vertex_np(filename):
    mesh = trimesh.load(filename)
    v = mesh.vertices
    return np.array(v, dtype=np.float32)


def load_vertex_torch(filename):
    mesh = trimesh.load(filename)
    v = mesh.vertices
    return torch.tensor(v, dtype=torch.float32)


def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


def read_ply_color(filename):
    """ read XYZRGBA point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z, r, g, b, a] for x, y, z, r, g, b, a in pc])
    return pc_array


def ply_to_glb(path):
    save = path.replace('.ply', '.glb')
    scene = a3d.Scene.from_file(path)
    scene.save(save)


def normalize_point_cloud(pc):
    if isinstance(pc, np.ndarray):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc_normalized = pc / m
    elif isinstance(pc, torch.Tensor):
        centroid = torch.mean(pc, dim=0)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc**2, dim=1)))
        pc_normalized = pc / m
    else:
        raise ValueError('point cloud must be np.ndarray or torch.Tensor')
    return pc_normalized, centroid, m


def compute_chamfer_distance(path_source, path_target, deivce='cpu', norm=False):

    pc_src = load_vertex_torch(path_source).to(deivce)
    pc_tgt = load_vertex_torch(path_target).to(deivce)

    if norm == True:
        pc_src, _, _ = normalize_point_cloud(pc_src)
        pc_tgt, _, _ = normalize_point_cloud(pc_tgt)

    cd, _ = chamfer_distance(pc_src[None, ...], pc_tgt[None, ...])

    return cd
