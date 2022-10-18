import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from modulus.hydra import to_absolute_path
from modulus.utils.io.vtk import var_to_polyvtk
from modulus.utils.io import InferencerPlotter


def generate_velocity_profile_3d():
    data = np.load(to_absolute_path("outputs/toy_car/inferencers/simulation.npz"), allow_pickle=True)
    data = np.atleast_1d(data.f.arr_0)[0]
    # velocity in 3D
    pos = np.dstack((data["x"], data["y"], data["z"]))
    V = np.dstack((data["u"], data["v"], data["w"]))

    save_var = {
        "x": data["x"],
        "y": data["y"],
        "z": data["z"],
        "p": data["p"],
        "pos": pos,
        "V": V,
    }

    var_to_polyvtk(save_var, to_absolute_path("outputs/toy_car/inferencers/velocity_profile"))


class InferencerSlicePlotter2D(InferencerPlotter):
    "Default plotter class for inferencer"

    def __call__(self, invar, outvar):
        "Default function for plotting inferencer data"

        # get input variables
        x, y = invar["x"][:, 0], invar["y"][:, 0]
        bounds = (x.min(), x.max(), y.min(), y.max())

        extent, outvar = self.interpolate_output(100, x, y, bounds, outvar)

        # make plots
        fs = []
        for k in outvar:
            f = plt.figure(figsize=(5, 4), dpi=144)
            plt.imshow(outvar[k].T, origin="lower", extent=extent, cmap="jet")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.colorbar(location="bottom")
            plt.title(k)
            plt.tight_layout()
            fs.append((f, k))

        return fs

    @staticmethod
    def interpolate_output(size, x, y, extent, *outvars):
        "Interpolates irregular points onto a mesh"

        # define mesh to interpolate onto
        xyi = np.meshgrid(
            np.linspace(extent[0], extent[1], size),
            np.linspace(extent[2], extent[3], size),
            indexing="ij",
        )

        # interpolate outvars onto mesh
        outvars_interp = []
        for outvar in outvars:
            outvar_interp = {}
            for k in outvar:
                outvar_interp[k] = scipy.interpolate.griddata(
                    (x, y), outvar[k][:, 0], tuple(xyi)
                )
            outvars_interp.append(outvar_interp)

        return [extent] + outvars_interp
