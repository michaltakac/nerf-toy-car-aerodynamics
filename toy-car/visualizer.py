import omni.usd
import omni.timeline

# Import the HPC visualization pipeline
from hpcvis.vtkm_bridge.core import get_bridge_interface
import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdUtils

import types
from dataclasses import dataclass
from typing import List

from .constants import bounds

# Put interface object publicly to use in our API
_vtkm_bridge = None


class VisParameters:
    def __init__(self):
        self.bounds = np.array(bounds).flatten()
        self.isovalue = 0.001
        self.streamline_count = 200
        self.streamline_step_size = 0.01
        self.streamline_step_count = 750
        self.streamline_radius = 0.02
        self.streamline_height = 0.0
        self._slice_x_pos = 0.5
        self._slice_y_pos = 0.5
        self._slice_z_pos = 0.5

    @property
    def slice_x_pos(self):
        return self._slice_x_pos

    @slice_x_pos.setter
    def slice_x_pos(self, offset):
        self._slice_x_pos = max(
            min((offset - self.bounds[0]) / (self.bounds[1] - self.bounds[0]), 1), 0
        )

    @property
    def slice_y_pos(self):
        return self._slice_y_pos

    @slice_y_pos.setter
    def slice_y_pos(self, offset):
        self._slice_y_pos = max(
            min((offset - self.bounds[2]) / (self.bounds[3] - self.bounds[2]), 1), 0
        )

    @property
    def slice_z_pos(self):
        return self._slice_z_pos

    @slice_z_pos.setter
    def slice_z_pos(self, offset):
        self._slice_z_pos = max(
            min((offset - self.bounds[4]) / (self.bounds[5] - self.bounds[4]), 1), 0
        )


class Visualizer:
    def __init__(self):
        # Get the vtkm bridge context
        self._vtkm_bridge = None
        print(
            f"[modulus_ext.scenario.toy_car.visualizer]_vtkm_bridge interface: {self._vtkm_bridge}"
        )
        self.parameters = VisParameters()

        self.velocity = None
        self.all_points = None
        self.bounds = None

        self._stage_id = None
        self._isosurface_primname = None
        self._streamlines_primname = None
        self._slice_primname_x = None
        self._slice_primname_y = None
        self._slice_primname_z = None

        self._seedpoints = None
        self._usd_context = None

        self.timeline = omni.timeline.acquire_timeline_interface()

    def get_geometry_prim(self, bridge_prim_name: str):
        stage = self._usd_context.get_stage()
        new_suffix = "_geometry"
        prim_name = bridge_prim_name.rsplit("_", maxsplit=1)[0] + new_suffix
        return stage.GetPrimAtPath(f"/RootClass/geometries/{prim_name}")

    def focus_prim(self, prim: Usd.Prim):
        if not prim.IsValid():
            return
        self._usd_context.get_selection().set_selected_prim_paths(
            [str(prim.GetPath())], True
        )
        try:
            import omni.kit.viewport_legacy

            viewport = omni.kit.viewport_legacy.get_viewport_interface()
            if viewport:
                viewport.get_viewport_window().focus_on_selected()
        except:
            raise
            pass

    def update_data(
        self,
        velocity: np.ndarray,
        velmag: np.ndarray,
        bounds: List[int],
        mask: np.ndarray = None,
        resolution: List[int] = [190, 190, 190],
    ):
        self.velocity = velocity
        self.bounds = bounds
        self.velmag = velmag
        
        def nan_ptp(a):
            return np.ptp(a[np.isfinite(a)])
        self.velmag = (self.velmag - np.nanmin(self.velmag))/nan_ptp(self.velmag)


        coords_x = np.linspace(self.bounds[0], self.bounds[1], resolution[0])
        coords_y = np.linspace(self.bounds[2], self.bounds[3], resolution[1])
        coords_z = np.linspace(self.bounds[4], self.bounds[5], resolution[2])
        Z, Y, X = np.meshgrid(coords_z, coords_y, coords_x, indexing="ij")

        self.all_points = np.array(
            np.transpose([C.flatten() for C in [X, Y, Z]]),
            copy=True,
            order="C",
            dtype=np.float32,
        )

        duplicated_velmag = np.expand_dims(self.velmag, axis=-1)
        np.seterr(invalid="ignore")
        self.normalized_velocity = self.velocity / duplicated_velmag

        #self.normalized_velocity = self.velocity  / np.amax(self.velocity)
        self.normalized_velocity[mask] = 0

        self.update_stage()
        self._vtkm_bridge.set_field_data("toy_car_velocity", velocity, n_components=3)
        self._vtkm_bridge.set_field_data(
            "toy_car_normalized_velocity", self.normalized_velocity, n_components=3
        )
        self._vtkm_bridge.set_field_data("toy_car_velmag", velmag, n_components=1)
        self._vtkm_bridge.set_regular_grid_bounds("toy_car", *bounds)
        self._vtkm_bridge.set_regular_grid_extent(
            "toy_car", *tuple(reversed(velmag.shape[:3]))
        )
        if self._seedpoints is not None:
            self._vtkm_bridge.set_points("toy_car_points", self._seedpoints)

        self.update_generated()
        

    def update_generated(self):
        if self._isosurface_primname:
            self.generate_isosurface()
        if self._streamlines_primname:
            self.generate_streamlines()
        if self._slice_primname_x or self._slice_primname_y or self._slice_primname_z:
            self.generate_slices()

    def update_stage(self):
        if self._vtkm_bridge is None:
            self._vtkm_bridge = get_bridge_interface()
        # Use the bridge to generate an isosurface on the data
        if self._usd_context is None:
            self._usd_context = omni.usd.get_context()
        stage = self._usd_context.get_stage()
        stage_cache = UsdUtils.StageCache.Get()
        stage_id = stage_cache.GetId(stage).ToLongInt()
        if stage_id == self._stage_id:
            return
        self._stage_id = stage_id
        self._vtkm_bridge.set_stage(stage_id)

    def random_subset(self, points, values, npoints=25):
        nonzero_selection = self.velmag.ravel() > 0.001 # Only points with some velocity

        points_nonzero = points[nonzero_selection]
        velmag_nonzero = self.velmag.ravel()[nonzero_selection]
        print(f"points_nonzero: {points_nonzero[:10]}")
        print(f"velmag_nonzero: {velmag_nonzero[:10]}")

        points_nonzero_shuffle = np.random.shuffle(points_nonzero)
        points_subset = points_nonzero[:npoints]
        velmag_subset = velmag_nonzero[:npoints]
        return points_subset

    def generate_streamlines(self):
        self.update_stage()
        # Use the bridge to generate streamlines on the data
        np.random.seed(42)
        self._seedpoints = self.random_subset(
            self.all_points, self.velocity, npoints=self.parameters.streamline_count
        )

        self._vtkm_bridge.set_points("toy_car_points", self._seedpoints)
        temp = self._streamlines_primname
        self._streamlines_primname = self._vtkm_bridge.visualize_streamlines(
            enabled=True,
            streamline_name="toy_car_streamlines",
            velocity_grid_name="toy_car",
            velocity_data_array_name="toy_car_normalized_velocity",
            sample_quantity_name="toy_car_velmag",
            seed_points_name="toy_car_points",
            step_size=self.parameters.streamline_step_size,
            n_steps=int(self.parameters.streamline_step_count),
            enable_tube_filter=True,
            tube_radius=self.parameters.streamline_radius,
        )

        if not self._streamlines_primname:
            print("Problem with streamline generation. Keeping old primname.")
            self._streamlines_primname = temp

        print(f"visualized streamlines: {self._streamlines_primname}")
        if not temp and self._streamlines_primname:
            prim = self.get_geometry_prim(self._streamlines_primname)
            self.focus_prim(prim)

        self.timeline.set_end_time(10)

    def generate_isosurface(self):
        self.update_stage()

        # velocity magnitude isosurface
        isosurface_prim = self._vtkm_bridge.visualize_isosurface(
            enabled=True,
            isosurface_name="toy_car_isosurface",
            regular_grid_name="toy_car",
            field_data_name="toy_car_velmag",
            sample_quantity_name="toy_car_velmag",
            isovalue=self.parameters.isovalue,
        )

        print(f"visualized isosurface: {self._isosurface_primname}")
        if not self._isosurface_primname:
            print("Problem with isosurface generation. Keeping old primname.")
            self._isosurface_primname = isosurface_prim

        if not isosurface_prim and self._isosurface_primname:
            prim = self.get_geometry_prim(self._isosurface_primname)
            self.focus_prim(prim)

    def generate_slices(self):
        self.update_stage()

        temp_x = self._slice_primname_x
        temp_y = self._slice_primname_y
        temp_z = self._slice_primname_z

        # Use the bridge to generate slices for the data
        self._slice_primname_x = self._vtkm_bridge.visualize_slice(
            enabled=True,
            slice_name="toy_car_slice_x",
            regular_grid_name="toy_car",
            field_data_name="toy_car_velmag",
            az=90.,
            el=0.0,
            pos=self.parameters.slice_x_pos,
        )
        print(f"visualized slice: {self._slice_primname_x}")
        self._slice_primname_y = self._vtkm_bridge.visualize_slice(
            enabled=True,
            slice_name="toy_car_slice_y",
            regular_grid_name="toy_car",
            field_data_name="toy_car_velmag",
            az=0.0,
            el=0.0,
            pos=self.parameters.slice_y_pos,
        )
        print(f"visualized slice: {self._slice_primname_y}")
        self._slice_primname_z = self._vtkm_bridge.visualize_slice(
            enabled=True,
            slice_name="toy_car_slice_z",
            regular_grid_name="toy_car",
            field_data_name="toy_car_velmag",
            az=0.0,
            el=90,
            pos=self.parameters.slice_z_pos,
        )
        print(f"visualized slice: {self._slice_primname_z}")

        if not self._slice_primname_x:
            print("Problem with slice generation. Keeping old primname.")
            self._slice_primname_x = temp_x

        if not self._slice_primname_y:
            print("Problem with slice generation. Keeping old primname.")
            self._slice_primname_y = temp_y

        if not self._slice_primname_z:
            print("Problem with slice generation. Keeping old primname.")
            self._slice_primname_z = temp_z

        if not temp_z and self._slice_primname_z:
            prim = self.get_geometry_prim(self._slice_primname_z)
            self.focus_prim(prim)
