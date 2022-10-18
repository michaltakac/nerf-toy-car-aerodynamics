from sympy import Symbol, Eq
from modulus.hydra import instantiate_arch, ModulusConfig
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.key import Key
from modulus.eq.pdes.navier_stokes import NavierStokes
from modulus.eq.pdes.turbulence_zero_eq import ZeroEquation
from modulus.eq.pdes.basic import NormalDotVec

# params for simulation
#############
# Real Params
#############
# fluid params
fluid_viscosity = 1.84e-05  # kg/m-s
fluid_density = 1.1614  # kg/m3

# boundary params
length_scale = 0.04  # m
inlet_velocity = 5.24386  # m/s

##############################
# Nondimensionalization Params
##############################
# fluid params
nu = fluid_viscosity / (fluid_density * inlet_velocity * length_scale)
rho = 1
normalize_inlet_vel = 1.0

# heat params
D_solid = 0.1
D_fluid = 0.02
inlet_T = 0
source_grad = 1.5
source_area = source_dim[0] * source_dim[2]

u_profile = (
    normalize_inlet_vel
    * tanh((0.5 - Abs(y)) / 0.02)
    * tanh((0.5625 - Abs(z)) / 0.02)
)
volumetric_flow = 1.0668  # value via integration of inlet profile

inlet_vel = Symbol("inlet_velocity")
x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

# parameterization
inlet_vel_range = (0.05, 10.0)
inlet_vel_params = {inlet_vel: inlet_vel_range}


def network(cfg: ModulusConfig, scale):
    # make list of nodes to unroll graph on
    ze = ZeroEquation(nu=nu, dim=3, time=False, max_distance=0.5)
    ns = NavierStokes(nu=ze.equations["nu"], rho=1.0, dim=3, time=False)
    normal_dot_vel = NormalDotVec(["u", "v", "w"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("inlet_velocity")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    return (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network", jit=cfg.jit)]
    )


def constraints(cfg: ModulusConfig, geo, nodes, domain):
    # add constraints to solver

    # inlet
    constraint_inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo.inlet,
        outvar={"u": u_profile, "v": 0, "w": 0},
        batch_size=cfg.batch_size.inlet,
        criteria=Eq(x, channel_origin[0]),
        lambda_weighting={"u": 1.0, "v": 1.0, "w": 1.0},
        batch_per_epoch=5000,
    )
    domain.add_constraint(constraint_inlet, "inlet")

    # outlet
    constraint_outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        criteria=Eq(x, channel_origin[0] + channel_dim[0]),
        batch_per_epoch=5000,
    )
    domain.add_constraint(constraint_outlet, "outlet")


    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo.noslip_mesh,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.no_slip,
        parameterization=inlet_vel_params
    )
    domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo.interior_mesh,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=inlet_vel_params
    )
    domain.add_constraint(interior, "interior")

    # integral continuity
    def integral_criteria(invar, params):
        sdf = geo.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_plane,
        outvar={"normal_dot_vel": volumetric_flow},
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.integral_continuity,
        criteria=integral_criteria,
        lambda_weighting={"normal_dot_vel": 1.0},
        parameterization={**x_pos_range, **param_ranges},
        batch_per_epoch=5000,
    )
    flow_domain.add_constraint(integral_continuity, "integral_continuity")


    # add pressure monitor
    invar_front_pressure = integral_plane.sample_boundary(
        1024,
        parameterization={
            x_pos: heat_sink_base_origin[0] - 0.65,
            **fixed_param_ranges,
        },
    )
    pressure_monitor = PointwiseMonitor(
        invar_front_pressure,
        output_names=["p"],
        metrics={"front_pressure": lambda var: torch.mean(var["p"])},
        nodes=flow_nodes,
    )
    domain.add_monitor(pressure_monitor)
    invar_back_pressure = integral_plane.sample_boundary(
        1024,
        parameterization={
            x_pos: heat_sink_base_origin[0] + 2 * 0.65,
            **fixed_param_ranges,
        },
    )
    pressure_monitor = PointwiseMonitor(
        invar_back_pressure,
        output_names=["p"],
        metrics={"back_pressure": lambda var: torch.mean(var["p"])},
        nodes=flow_nodes,
    )
    domain.add_monitor(pressure_monitor)

