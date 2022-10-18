import os
import torch
import shutil
import asyncio
import traceback
import omni.ext
import omni.usd
import omni.ui as ui
import numpy as np

from pathlib import Path
from modulus.hydra.utils import compose

from modulus_ext.ui.scenario import (
    ModulusOVScenario,
    ModulusOVButton,
    ModulusOVFloatSlider,
    ModulusOVIntSlider,
    ModulusOVToggle,
    ModulusOVRow,
    ModulusOVText,
    ModulusOVProgressBar,
)

from .visualizer import Visualizer
from .toy_car_runner import ModulusToyCarRunner
from .constants import bounds
from .src.toy_car import inlet_vel_range

class ToyCarScenario(ModulusOVScenario):
    def __init__(self):
        self._init_task = asyncio.ensure_future(self.deferred_init())

    async def deferred_init(self):
        super().__init__(name="Toy car aerodynamics simulator Omniverse Extension")
        
        # Need to be a few frames in before init can occur.
        # This is required for auto-loading of the extension
        for i in range(15):
            await omni.kit.app.get_app().next_update_async()
        
        self.solver_train_initialized = False
        self.solver_eval_initialized = False
        self._eval_complete = False
        self.resolution = [128, 128, 128]

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 10**9
        eco = vram_gb < 13  # 12 Gb and below GPUs, turn on eco mode

        self.inlet_velocity = 1.5

        self.visualizer = Visualizer()
        self._usd_context = omni.usd.get_context()

        if self._usd_context.is_new_stage():
            self.load_template()

        param_text = ModulusOVText(
            desc="Input Parameters",
        )
        self.add(param_text)
        height_slider = ModulusOVFloatSlider(
            name="Inlet Velocity",
            desc="Inlet velocity from the top for Inference",
            default_value=self.inlet_velocity,
            bounds=inlet_vel_range,
            update_func=self.update_inlet_velocity,
        )
        self.add(height_slider)

        # Inference controls
        self.inf_button = ModulusOVButton(
            name="Inference",
            desc="Perform Inference",
            update_func=self.run_inference,
        )
        self.inf_button.run_in_main_thread = False
        self.add(self.inf_button)

        self.inf_progress = ModulusOVProgressBar(
            desc="Inference Progress", default_value=0.0
        )
        self.inf_progress.inference_scale = 0.7
        self.add(self.inf_progress)

        # Visualization actions
        isosurfaceButton = ModulusOVButton(
            name="Isosurface",
            desc="Generate Isosurface Visualization",
            update_func=self.generate_isosurface,
        )
        streamlineButton = ModulusOVButton(
            name="Streamline",
            desc="Generate Streamline Visualization",
            update_func=self.generate_streamlines,
        )
        sliceButton = ModulusOVButton(
            name="Slice",
            desc="Generate Slice Visualization",
            update_func=self.generate_slices,
        )
        button_row = ModulusOVRow(
            elements=[isosurfaceButton, streamlineButton, sliceButton]
        )
        self.add(button_row)

        # Isosuface controls
        control_text = ModulusOVText(
            desc="Isosurface Controls",
        )
        self.add(control_text)
        slider = ModulusOVFloatSlider(
            name="Isovalue",
            desc="Isosurface visualization isovalue",
            default_value=0.001,
            bounds=(0.001, 1.0),
            update_func=self.update_isovalue,
        )
        self.add(slider)

        # Streamline controls
        control_text = ModulusOVText(
            desc="Streamline Controls",
        )
        self.add(control_text)
        slider = ModulusOVIntSlider(
            name="Streamline Count",
            desc="Streamline visualization count",
            default_value=200,
            bounds=(1, 400),
            update_func=self.update_streamline_count,
        )
        self.add(slider)

        slider = ModulusOVFloatSlider(
            name="Streamline Step Size",
            desc="Step Size used for Calculating Streamlines",
            default_value=0.01,
            bounds=(0.001, 0.1),
            update_func=self.update_streamline_step_size,
        )
        self.add(slider)

        slider = ModulusOVIntSlider(
            name="Streamline Step Count",
            desc="Number of Integration Steps to Calculate Streamlines",
            default_value=1000,
            bounds=(1, 2000),
            update_func=self.update_streamline_step_count,
        )
        self.add(slider)

        slider = ModulusOVFloatSlider(
            name="Streamline Radius",
            desc="Radius of Streamline Tubes",
            default_value=0.02,
            bounds=(0.0001, 0.1),
            update_func=self.update_streamline_radius,
        )
        self.add(slider)

        # Slice controls
        control_text = ModulusOVText(
            desc="Slice Controls",
        )
        self.add(control_text)
        slider = ModulusOVFloatSlider(
            name="Slice X Offset",
            desc="Contour slice X offset from domain center",
            default_value=0.0,
            bounds=[bounds[0][0], bounds[0][1]],
            update_func=self.update_slice_x_offset,
        )
        self.add(slider)

        slider = ModulusOVFloatSlider(
            name="Slice Y Offset",
            desc="Contour slice Y offset from domain center",
            default_value=0.0,
            bounds=[bounds[1][0], bounds[1][1]],
            update_func=self.update_slice_y_offset,
        )
        self.add(slider)

        slider = ModulusOVFloatSlider(
            name="Slice Z Offset",
            desc="Contour slice Z offset from domain center",
            default_value=0.0,
            bounds=[bounds[2][0], bounds[2][1]],
            update_func=self.update_slice_z_offset,
        )
        self.add(slider)

        eco_toggle = ModulusOVToggle(
            name="Eco Mode",
            desc="For cards with limited memory",
            default_value=eco,
            update_func=self.toggle_eco,
        )
        self.add(eco_toggle)
        self.register()

        cfg = compose(config_name="config", config_path="conf", job_name="ToyCar")
        self.simulator_runner = ModulusToyCarRunner(
            cfg, progress_bar=self.inf_progress
        )
        self.simulator_runner.eco = eco

    def load_template(self):
        print("loading template")
        usd_context = omni.usd.get_context()
        template_file = Path(os.path.dirname(__file__)) / Path(
            "../data/toy_car_template.usda"
        )
        self.template_temp_file = str(
            Path(os.path.dirname(__file__))
            / Path("../data/toy_car_template_temp.usda")
        )
        shutil.copyfile(template_file, self.template_temp_file)
        usd_context.open_stage(self.template_temp_file)

    def toggle_eco(self, value):
        print(f"Eco mode set to {value}")
        self.simulator_runner.eco = value

    def run_inference(self):
        self.inf_button.text = "Running Inference..."
        print("Toy car simulator inferencer started")

        if self.simulator_runner.eco:
            resolution_x = 64
            resolution_y = 32
            resolution_z = 64
        else:
            resolution_x = 128
            resolution_y = 128
            resolution_z = 128

        if (resolution_x, resolution_y, resolution_z) != self.resolution:
            print(
                f"Initializing inferencer with a resolution of {resolution_x}*{resolution_y}*{resolution_z}"
            )
            self.resolution = [resolution_x, resolution_y, resolution_z]

        print(
            f"Will run inferencing for inlet_velocity={self.inlet_velocity}"
        )

        pred_vars = self.simulator_runner.run_inference(
            inlet_velocity=self.inlet_velocity,
            resolution=list(self.resolution),
        )

        shape = tuple(self.resolution)
        u = pred_vars["u"].reshape(shape)
        v = pred_vars["v"].reshape(shape)
        w = pred_vars["w"].reshape(shape)
        velocity = np.stack([u, v, w], axis=-1)
        if velocity.dtype != np.float32:
            velocity = velocity.astype(np.float32)

        if velocity.shape != shape + (3,):
            raise RuntimeError(f"expected shape: {shape + (3,)}; got: {velocity.shape}")
        # Change to z axis first for VTK input (not sure why)
        # Tensor comes out of inferencer in ij index form
        velocity = np.ascontiguousarray(velocity.transpose(2, 1, 0, 3))

        self.inf_progress.value = 0.95

        np.seterr(invalid="ignore")

        mask = np.where(velocity == self.simulator_runner.mask_value)
        velocity[mask] = 0.0
        velmag = np.linalg.norm(velocity, axis=3)
        # velmag = velmag / np.amax(velmag)
        minval = np.amin(velmag)
        maxval = np.amax(velmag)
        print("Test", maxval, minval)

        self._velocity = velocity
        self._velmag = velmag
        # self._mask = spatial_mask
        self._vel_mask = mask
        self._bounds = np.array(self.simulator_runner.bounds).flatten()

        print("ToyCarScenario inference ended")
        self._eval_complete = True
        self.inf_progress.value = 1.0
        self.inf_button.text = "Inference"
    def update_vis_data(self):
        if not all(v is not None for v in [self._velocity, self._velmag, self._bounds]):
            return
        self.visualizer.update_data(
            self._velocity, self._velmag, self._bounds, self._vel_mask, self.resolution
        )

    def update_inlet_velocity(self, value: float):
        self.inlet_velocity = value

    def update_isovalue(self, isovalue):
        print(f"Updating isovalue: {isovalue}")
        self.visualizer.parameters.isovalue = isovalue
        self.visualizer.update_generated()

    def update_streamline_count(self, streamline_count):
        print(f"Updating streamline_count: {streamline_count}")
        self.visualizer.parameters.streamline_count = streamline_count
        self.visualizer.update_generated()

    def update_streamline_step_size(self, streamline_step_size):
        print(f"Updating streamline_step_size: {streamline_step_size}")
        self.visualizer.parameters.streamline_step_size = streamline_step_size
        self.visualizer.update_generated()

    def update_streamline_step_count(self, streamline_step_count):
        print(f"Updating streamline_step_count: {streamline_step_count}")
        self.visualizer.parameters.streamline_step_count = streamline_step_count
        self.visualizer.update_generated()

    def update_streamline_radius(self, streamline_radius):
        print(f"Updating streamline_radius: {streamline_radius}")
        self.visualizer.parameters.streamline_radius = streamline_radius
        self.visualizer.update_generated()

    def update_slice_x_offset(self, slice_x_offset):
        print(f"Updating slice_x_offset: {slice_x_offset}")
        self.visualizer.parameters.slice_x_pos = slice_x_offset
        self.visualizer.update_generated()

    def update_slice_y_offset(self, slice_y_offset):
        print(f"Updating slice_y_offset: {slice_y_offset}")
        self.visualizer.parameters.slice_y_pos = slice_y_offset
        self.visualizer.update_generated()

    def update_slice_z_offset(self, slice_z_offset):
        print(f"Updating slice_z_offset: {slice_z_offset}")
        self.visualizer.parameters.slice_z_pos = slice_z_offset
        self.visualizer.update_generated()

    def generate_isosurface(self):
        if not self._eval_complete:
            print("Need to run inferencer first!")
            return
        self.update_vis_data()
        self.visualizer.generate_isosurface()

    def generate_streamlines(self):
        if not self._eval_complete:
            print("Need to run inferencer first!")
            return
        self.update_vis_data()
        self.visualizer.generate_streamlines()

    def generate_slices(self):
        if not self._eval_complete:
            print("Need to run inferencer first!")
            return
        self.update_vis_data()
        self.visualizer.generate_slices()


# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class ToyCarExt(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("[modulus.scenario.ToyCar] Toy car aerodynamics scenario startup")
        self.scenario = ToyCarScenario()

    def on_shutdown(self):
        self.scenario.__del__()
        print("[modulus.scenario.ToyCar] Toy car aerodynamics scenario shutdown")
