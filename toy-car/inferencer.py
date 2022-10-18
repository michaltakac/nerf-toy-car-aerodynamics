from modulus.hydra import to_yaml
from modulus.hydra.utils import compose
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.domain.inferencer import PointwiseInferencer, VoxelInferencer

from src.geometry import ToyCarDomain
from src.toy_car import network, constraints, inlet_vel
from src.plotter import generate_velocity_profile_3d, InferencerSlicePlotter2D


cfg = compose(config_path="conf", config_name="config_eval", job_name="toy_car_inference")
print(to_yaml(cfg))


def run():
    geo = ToyCarDomain()
    domain = Domain()
    nodes = network(cfg, scale=geo.scale)
    constraints(cfg, geo=geo, nodes=nodes, domain=domain)

    inlet_vel_inference = 0.1

    inferencer = PointwiseInferencer(
        nodes=nodes,
        invar=geo.interior_mesh.sample_interior(1000000, parameterization={inlet_vel: inlet_vel_inference}),
        output_names=["u", "v", "w", "p"],
        batch_size=1024,
        requires_grad=False,
        plotter=InferencerSlicePlotter2D()
    )
    domain.add_inferencer(inferencer, "simulation")
    
    # add meshgrid inferencer
    mask_fn = lambda x, y, z: geo.interior_mesh.sdf({"x": x, "y": y, "z": z})[0] < 0
    voxel_inference = VoxelInferencer(
        bounds=[[-3, 3], [-3, 3], [-3, 3]],
        npoints=[128, 128, 128],
        nodes=nodes,
        output_names=["u", "v", "w", "p"],
        export_map={"u": ["u", "v", "w"], "p": ["p"]},
        mask_fn=mask_fn,
        batch_size=1024,
        requires_grad=False,
    )
    domain.add_inferencer(voxel_inference, "simulation_voxel")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

    # generate velocity profile with magnitude (it has V = [u, v, w] in one array)
    generate_velocity_profile_3d()


if __name__ == "__main__":
    run()
