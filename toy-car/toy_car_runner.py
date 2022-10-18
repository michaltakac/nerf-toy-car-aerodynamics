import sys, os
import torch
import modulus
from sympy import Symbol, Eq, Abs, tanh
import numpy as np
import logging
from typing import List, Dict, Union
from pathlib import Path


from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.models.fully_connected import FullyConnectedArch
from modulus.domain.inferencer import (
    OVVoxelInferencer,
)
from modulus_ext.ui.scenario import ModulusOVProgressBar
from modulus.key import Key
from modulus.key import Key
from modulus.eq.pdes.navier_stokes import NavierStokes
from modulus.eq.pdes.basic import NormalDotVec
from modulus.geometry.tessellation import Tessellation

from .constants import bounds
from .src.geometry import ToyCarDomain


class ModulusToyCarRunner(object):

    """Toy car simulator inference runner for OV scenario

    Args:
        cfg (ModulusConfig): Parsed Modulus config
    """

    def __init__(
        self,
        cfg: ModulusConfig,
        progress_bar: ModulusOVProgressBar,
        mask_value: float = -100,
    ):

        logging.getLogger().addHandler(logging.StreamHandler())
        ##############################
        # Nondimensionalization Params
        ##############################
        # fluid params
        # Water at 20°C (https://wiki.anton-paar.com/en/water/)
        # https://en.wikipedia.org/wiki/Viscosity#Kinematic_viscosity
        self.nu = 1.787e-06  # m2 * s-1
        self.inlet_vel = Symbol("inlet_velocity")
        self.rho = 1
        self.scale = 1.0

        self.cfg = cfg
        self.progress_bar = progress_bar
        self._eco = False
        self._inferencer = None
        self.bounds = bounds
        self.mask_value = mask_value

    @property
    def eco(self):
        return self._eco

    @eco.setter
    def eco(self, value: bool):
        self._eco = value
        if self._inferencer:
            self._inferencer.eco = value

    def load_inferencer(self, checkpoint_dir: Union[str, None] = None):
        """Create Modulus Toy Car simulator inferencer object. This can take time since
        it will initialize the model

        Parameters
        ----------
        checkpoint_dir : Union[str, None], optional
            Directory to modulus checkpoint
        """
        # make list of nodes to unroll graph on
        ns = NavierStokes(nu=self.nu * self.scale, rho=self.rho, dim=3, time=False)
        normal_dot_vel = NormalDotVec(["u", "v", "w"])

        self.progress_bar.value = 0.025
        equation_nodes = (
            ns.make_nodes()
            + normal_dot_vel.make_nodes()
        )

        # determine inputs outputs of the network
        input_keys = [Key("x"), Key("y"), Key("z")]
        input_keys += [Key("inlet_velocity")]
        output_keys = [Key("u"), Key("v"), Key("w"), Key("p")]

        # select the network and the specific configs
        flow_net = FullyConnectedArch(
            input_keys=input_keys,
            output_keys=output_keys,
        )
        self.flow_nodes = equation_nodes + [
            flow_net.make_node(name="flow_network", jit=self.cfg.jit)
        ]

        invar_keys = [
            Key.from_str("x"),
            Key.from_str("y"),
            Key.from_str("z"),
            Key.from_str("inlet_velocity"),
        ]
        outvar_keys = [
            Key.from_str("u"),
            Key.from_str("v"),
            Key.from_str("w"),
            Key.from_str("p"),
        ]

        self._inferencer = OVVoxelInferencer(
            nodes=self.flow_nodes,
            input_keys=invar_keys,
            output_keys=outvar_keys,
            mask_value=self.mask_value,
            requires_grad=False,
            eco=False,
            progress_bar=self.progress_bar,
        )

        # Load checkpointed model
        if checkpoint_dir is not None:
            absolute_checkpoint_dir = Path(__file__).parent / checkpoint_dir
            if absolute_checkpoint_dir.resolve().is_dir():
                self._inferencer.load_models(absolute_checkpoint_dir.resolve())
            else:
                print("Could not find checkpointed model")
        # Set eco
        self._inferencer.eco = self.eco

    def load_geometry(self):
        # normalize meshes
        def normalize_mesh(mesh, center, scale):
            mesh = mesh.translate([-c for c in center])
            mesh = mesh.scale(scale)
            return mesh

        stl_path = Path(self.data_path) / Path("stl_files")
        self.car_mesh = Tessellation.from_stl(
            Path(stl_path) / Path("toy_bmw.stl"), airtight=True
        )
        center = (0, 0, 0)
        scale = 1.0

        self.car_mesh = normalize_mesh(self.car_mesh, center, scale)

    def run_inference(
        self,
        inlet_velocity: float,
        resolution: List[int] = [256, 256, 256],
    ) -> Dict[str, np.array]:
        """Runs inference for toy car simulator

        Args:
            resolution (List[int], optional): Voxel resolution. Defaults to [256, 256, 256].

        Returns:
            Dict[str, np.array]: Predicted output variables
        """
        self.progress_bar.value = 0
        if self._inferencer is None:
            print("Loading toy car inferencer")
            self.load_inferencer(checkpoint_dir="./checkpoints")
            self.progress_bar.value = 0.05
            print("Loading toy car geometry")
            self.load_geometry()
            self.progress_bar.value = 0.1

        # Eco mode settings
        if self._inferencer.eco:
            batch_size = 512
            memory_fraction = 0.1
        else:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 10**9
            batch_size = int((vram_gb // 6) * 16 * 1024)
            memory_fraction = 1.0

        mask_fn = (
            lambda x, y, z: self.car_mesh.sdf({"x": x, "y": y, "z": z}, {})["sdf"]
            < 0
        )

        sp_array = np.ones((np.prod(resolution), 1))

        specific_params = {
            "inlet_velocity": inlet_velocity * sp_array,
        }

        # Set up the voxel sample domain
        self._inferencer.setup_voxel_domain(
            bounds=self.bounds,
            npoints=resolution,
            invar=specific_params,
            batch_size=batch_size,
            mask_fn=mask_fn,
        )
        self.progress_bar.value = 0.2
        # Perform inference
        invar, predvar = self._inferencer.query(memory_fraction)
        # TODO: Remove should be changed to inside inferencer
        self.progress_bar._prev_step = 0.0
        self.progress_bar.value = 0.9

        return predvar

    @property
    def data_path(self):
        data_dir = Path(os.path.dirname(__file__)) / Path("../data")
        return str(data_dir)
