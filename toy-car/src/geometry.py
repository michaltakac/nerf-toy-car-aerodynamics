import numpy as np
from sympy import sqrt, Max
from modulus.hydra import to_absolute_path
from modulus.geometry.tessellation import Tessellation
from modulus.geometry.primitives_3d import Box, Channel, Plane



class ToyCarDomain:
    """Toy car geometry inside channel"""

    def __init__(self):
        # read stl files to make geometry
        point_path = to_absolute_path("./stl_files")

        car_mesh = Tessellation.from_stl(
            point_path + "/toy_bmw.stl", airtight=True
        )

        # scale and normalize mesh and openfoam data
        self.center = (0, 0, 0)
        self.scale = 1.0
        self.car_mesh = self.normalize_mesh(car_mesh, self.center, self.scale)

        # geometry params for domain
        channel_origin = (-2.5, -0.5, -0.5625)
        channel_dim = (5.0, 1.0, 1.125)

        # channel
        channel = Channel(
            channel_origin,
            (
                channel_origin[0] + channel_dim[0],
                channel_origin[1] + channel_dim[1],
                channel_origin[2] + channel_dim[2],
            ),
        )


    # normalize meshes
    def normalize_mesh(self, mesh, center, scale):
        mesh = mesh.translate([-c for c in center])
        mesh = mesh.scale(scale)
        return mesh

