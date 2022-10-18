from modulus.hydra import to_yaml
from modulus.hydra.utils import compose
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.domain.inferencer import PointwiseInferencer

from src.geometry import ToyCarDomain
from src.toy_car import network, constraints, inlet_vel

cfg = compose(config_path="conf", config_name="config", job_name="toy_car_training")
print(to_yaml(cfg))


def run():
    geo = ToyCarDomain()
    domain = Domain()
    nodes = network(cfg, scale=geo.scale)
    constraints(cfg, geo=geo, nodes=nodes, domain=domain)

    inlet_vel_inference = 1.5

    # add inferencer
    inferencer = PointwiseInferencer(
        nodes=nodes,
        invar=geo.interior_mesh.sample_interior(1000000, parameterization={inlet_vel: inlet_vel_inference}),
        output_names=["u", "v", "w", "p"],
        batch_size=4096,
    )
    domain.add_inferencer(inferencer, "inf_data")

    inferencer = PointwiseInferencer(
        nodes=nodes,
        invar=geo.interior_mesh.sample_interior(5000, parameterization={inlet_vel: inlet_vel_inference}),
        output_names=["u", "v", "w", "p"],
        batch_size=256,
    )
    domain.add_inferencer(inferencer, "inf_data_small")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
