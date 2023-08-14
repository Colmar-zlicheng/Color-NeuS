# This code is borrow from https://github.com/quan-meng/gnerf.git
import cv2
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Agg')


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def get_nsamples(data_loader, N):
    x = []
    n = 0
    while n < N:
        x_next, _ = next(iter(data_loader))
        x.append(x_next)
        n += x_next.size(0)
    x = torch.cat(x, dim=0)[:N]
    return x


def get_camera_wireframe(scale: float = 0.03):
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    a = 0.5 * torch.tensor([-2, 1.5, 4])
    up1 = 0.5 * torch.tensor([0, 1.5, 4])
    up2 = 0.5 * torch.tensor([0, 2, 4])
    b = 0.5 * torch.tensor([2, 1.5, 4])
    c = 0.5 * torch.tensor([-2, -1.5, 4])
    d = 0.5 * torch.tensor([2, -1.5, 4])
    C = torch.zeros(3)
    F = torch.tensor([0, 0, 3])
    camera_points = [a, up1, up2, up1, b, d, c, a, C, b, d, C, c, C, F]
    lines = torch.stack([x.float() for x in camera_points]) * scale
    return lines


def plot_cameras(ax, c2w, color: str = "blue", scale=1.0, opengl=False):
    if opengl:
        y, z = -1, -1
    else:
        y, z = 1, 1
    device = c2w.device
    nbatch = c2w.shape[0]
    cam_wires_canonical = get_camera_wireframe(scale)[None].to(device)
    R = c2w[:, :3, :3] @ torch.tensor([[1., 0, 0], [0, y, 0], [0, 0, z]], device=device)
    R = torch.cat([r.T[None] for r in R], 0)
    cam_wires_trans = torch.bmm(cam_wires_canonical.repeat(nbatch, 1, 1), R) + c2w[:, None, :3, -1]
    plot_handles = []
    for wire in cam_wires_trans:
        # the Z and Y axes are flipped intentionally here!
        x_, y_, z_ = wire.detach().cpu().numpy().T.astype(float)
        (h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.3)
        plot_handles.append(h)
    return plot_handles


def fig2img(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def plot_camera_scene(c2w, c2w_gt=None, plot_radius=5.0, status='', opengl=False):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=45., azim=60)
    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([-plot_radius, plot_radius])
    ax.set_zlim3d([0, plot_radius * 2])

    xspan, yspan, zspan = 3 * [np.linspace(-plot_radius, plot_radius, 20)]
    zero = np.zeros_like(xspan)
    ax.plot3D(xspan, zero, zero, 'k--')
    ax.plot3D(zero, yspan, zero, 'k--')
    ax.plot3D(zero, zero, zspan + plot_radius, 'k--')
    ax.text(plot_radius, .5, .5, "x", color='red')
    ax.text(.5, plot_radius, .5, "y", color='green')
    ax.text(.5, .5, plot_radius * 2, "z", color='blue')

    scale = 0.05 * plot_radius
    handle_cam = plot_cameras(ax, c2w, color="#FF7D1E", scale=scale, opengl=opengl)
    if c2w_gt is not None:
        handle_cam_gt = plot_cameras(ax, c2w_gt, color="#812CE5", scale=scale, opengl=opengl)

        labels_handles = {
            "Estimated Cameras": handle_cam[0],
            "GT Cameras": handle_cam_gt[0],
        }
    else:
        labels_handles = {"Estimated cameras": handle_cam[0]}

    ax.legend(labels_handles.values(),
              labels_handles.keys(),
              loc="upper center",
              bbox_to_anchor=(0.32, 0.7),
              prop={'size': 8})

    ax.axis('off')
    fig.tight_layout()

    img = fig2img(fig)

    plt.close(fig)

    img = ToTensor()(img)

    return img


def plot_cameras_track(c2w):
    xyz = c2w[:, :3, 3]
    xyz = xyz.detach().cpu().numpy()

    fig = plt.figure()

    ax = fig.add_axes(Axes3D(fig))
    ax.set_title("3D_Curve")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    figure = ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='r', lw=0.5)

    img = fig2img(fig)
    plt.close(fig)

    img = ToTensor()(img)
    return img


# This code is borrow from https://github.com/sxyu/pixel-nerf.git
def image_float_to_uint8(img):
    """
    Convert a float image (0.0-1.0) to uint8 (0-255)
    """
    vmin = np.min(img)
    vmax = np.max(img)
    if vmax - vmin < 1e-10:
        vmax += 1e-10
    img = (img - vmin) / (vmax - vmin)
    img *= 255.0
    return img.astype(np.uint8)


def cmap(img, color_map=cv2.COLORMAP_HOT):
    """
    Apply 'HOT' color to a float image
    """
    return cv2.applyColorMap(image_float_to_uint8(img), color_map)
