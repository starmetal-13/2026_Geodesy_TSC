# GeoSlip2D

GeoSlip2D is a Python package for computing two-dimensional Green's functions for slip on a subduction-interface profile and using those Green's functions in 1D profile slip inversions.

The package is designed around a common workflow:

```text
interface geometry
      ↓
Green's-function backend
      ↓
Greens2D object
      ↓
save / load
      ↓
profile slip inversion
```

GeoSlip2D currently includes Green's-function tools for:

- homogeneous elastic half-space calculations,
- layered elastic models,
- compliant elastic wedge models,
- viscoelastic earthquake-cycle models,
- and profile slip inversions using horizontal and optionally vertical surface velocities.

The package is intended for testing how assumptions about elastic structure and earthquake-cycle processes affect inferred slip or slip-deficit distributions on a 2D dipping fault interface.

---

## Repository layout

A typical repository layout is:

```text
GeoSlip2D/
├── README.md
├── pyproject.toml
├── src/
│   └── geoslip2d/
│       ├── __init__.py
│       ├── geometry.py
│       ├── greens.py
│       ├── homogeneous.py
│       ├── layered.py
│       ├── wedge.py
│       ├── vecycle.py
│       ├── inversion.py
│       ├── io.py
│       ├── plotting.py
│       ├── elastic_wedge/
│       └── vecycle2d/
├── examples/
│   ├── Compliant_wedge_model.ipynb
│   ├── Layered_model.ipynb
│   ├── Slip_inversion.ipynb
│   └── Viscoelastic_cycle.ipynb
└── tests/
```

The `examples/` folder contains the main usage notebooks. These notebooks are the recommended entry point for new users.

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/<your-username>/GeoSlip2D.git
cd GeoSlip2D
pip install -e .
```

For a fresh environment, a typical setup is:

```bash
conda create -n geoslip2d python=3.11
conda activate geoslip2d
pip install -e .
```

GeoSlip2D uses standard scientific Python packages, including NumPy, SciPy, and Matplotlib.

---

## Core concepts

### InterfaceGeometry

The `InterfaceGeometry` object defines the dipping fault/interface geometry. It stores patch-end coordinates, patch centers, patch lengths, and metadata describing how the geometry was created.

A typical interface is created with:

```python
import geoslip2d as gs2d

interface = gs2d.make_interface_geometry_legacy(
    faultdip_trench=10.0,
    x_trench=0.0,
    x_bottom=238.0,
    faultdip_bottom=20.0,
    z_bottom=45.0,
    patch_length=5.0,
)

print(interface.summary())
```

GeoSlip2D uses a positive-down depth convention for its public interface geometry.

### Greens2D

All Green's-function backends return a common `Greens2D` object with fields such as:

```python
greens.Ghor              # horizontal Green's matrix
greens.Gvert             # vertical Green's matrix
greens.xobs              # observation coordinates
greens.interface         # InterfaceGeometry object
greens.source_type       # backend name
greens.metadata          # backend-specific metadata
```

The Green's matrices have shape:

```text
number of observation points × number of fault/interface patches
```

This common object makes it possible to use the same inversion code with homogeneous, layered, wedge, or viscoelastic-cycle Green's functions.

---

## Green's-function backends

### 1. Homogeneous elastic backend

The homogeneous backend computes surface displacement Green's functions for finite edge dislocations in an elastic half-space.

```python
import numpy as np
import geoslip2d as gs2d

interface = gs2d.make_interface_geometry_legacy(
    faultdip_trench=10.0,
    x_trench=0.0,
    x_bottom=238.0,
    faultdip_bottom=20.0,
    z_bottom=45.0,
    patch_length=5.0,
)

xobs = np.linspace(0.5, 500.0, 300)

greens = gs2d.build_greens(
    "homogeneous",
    interface,
    xobs,
    length_override=5.0,
    progress=True,
)

gs2d.save_greens("Greens_homogeneous.npz", greens)
```

The homogeneous backend is the simplest backend and is useful as a reference model for comparison with more complex Green's functions.

### 2. Layered elastic backend

The layered backend computes Green's functions for a layered elastic structure.

```python
import numpy as np
import geoslip2d as gs2d

interface = gs2d.make_interface_geometry_legacy(
    faultdip_trench=10.0,
    x_trench=0.0,
    x_bottom=238.0,
    faultdip_bottom=20.0,
    z_bottom=45.0,
    patch_length=5.0,
)

xobs = np.linspace(0.5, 500.0, 300)

layered_cfg = gs2d.LayeredConfig(
    h=np.array([5.0, 10.0, 15.0]),
    mu=np.array([1.0, 1.0, 1.0, 1.0]),
    nu=np.array([0.25, 0.25, 0.25, 0.25]),
    progress=True,
)

greens = gs2d.build_greens(
    "layered",
    interface,
    xobs,
    layered_cfg,
)

gs2d.save_greens("Greens_layered.npz", greens)
```

The layered backend uses the same `InterfaceGeometry` and `xobs` grid as the homogeneous backend, allowing direct comparisons.

### 3. Compliant wedge backend

The compliant wedge backend computes Green's functions for an elastic wedge geometry with different elastic domains. It is useful for testing how compliant wedge structure affects surface velocities and inferred slip.

```python
import numpy as np
import geoslip2d as gs2d

interface = gs2d.make_interface_geometry_legacy(
    faultdip_trench=3.0,
    x_trench=0.0,
    x_bottom=200.0,
    faultdip_bottom=20.0,
    z_bottom=50.0,
    patch_length=5.0,
)

xobs = np.linspace(0.5, 500.0, 300)

wedge_cfg = gs2d.WedgeConfig(
    wedge_bot=12.0,
    wedge_top_x=90.0,
    L_slab=50.0,
    W=200.0,
    mu1=1.0,
    mu2=1.0,
    mu3=1.0,
    nu=0.25,
    pL=5.0,
    progress=True,
    plot_geometry=True,
    plot_geometry_xlim=(0.0, 300.0),
)

greens = gs2d.build_greens(
    "wedge",
    interface,
    xobs,
    wedge_cfg,
)

gs2d.save_greens("Greens_wedge.npz", greens)
```

The wedge solver internally constructs a full boundary-element wedge geometry, but the returned `Greens2D` object is interpolated onto the user-specified `xobs` grid.

You can also plot the wedge geometry directly:

```python
fig, ax = gs2d.plot_wedge_geometry(
    config=wedge_cfg,
    interface=interface,
    xlim=(0.0, 300.0),
)
```

### 4. Viscoelastic-cycle backend

The viscoelastic-cycle backend computes Green's functions that include earthquake-cycle effects. It is useful for exploring how viscoelastic relaxation and periodic earthquake cycles affect interseismic velocities and inferred slip.

```python
import numpy as np
import geoslip2d as gs2d

interface = gs2d.make_interface_geometry_legacy(
    faultdip_trench=5.0,
    x_trench=0.0,
    x_bottom=300.0,
    faultdip_bottom=35.0,
    z_bottom=80.0,
    patch_length=8.0,
)

xobs = np.linspace(0.5, 500.0, 300)

vecycle_cfg = gs2d.VECycleConfig(
    mode="build",
    component="interseismic",
    progress=True,
)

greens = gs2d.build_greens(
    "vecycle",
    interface,
    xobs,
    vecycle_cfg,
)

gs2d.save_greens("Greens_vecycle.npz", greens)
```

Note: the current viscoelastic-cycle backend uses its own native geometry builder internally and then converts the output into a `Greens2D` object. The returned Green's functions are placed on the requested `xobs` grid. When comparing to the homogeneous backend, use the interface attached to the returned VECycle `Greens2D` object if exact source-geometry consistency is required.

---

## Saving and loading Green's functions

GeoSlip2D supports saving and loading Green's functions through a common interface.

```python
gs2d.save_greens("Greens_homogeneous.npz", greens)
greens2 = gs2d.load_greens("Greens_homogeneous.npz")
```

The native recommended format is `.npz`.

The saved `Greens2D` object includes the Green's matrices, observation coordinates, interface geometry, source type, sign convention, and metadata.

---

## Profile slip inversion

GeoSlip2D can invert 1D profile velocities for slip or slip-deficit rate on the interface.

### Direct inversion from projected profile observations

```python
import geoslip2d as gs2d

observations = gs2d.ProfileObservations(
    x_hor=xobs,
    v_hor=v_hor,
    sig_hor=sig_hor,
    x_vert=xobs,
    v_vert=v_vert,
    sig_vert=sig_vert,
)

inv_cfg = gs2d.SlipInversionConfig(
    alpha=1.0,
    smoothing_order="second",
    solver_type="nonnegative",
    inversion_mode="forward_slip",
    use_vertical=True,
)

out = gs2d.fit_profile_slip(
    observations,
    greens,
    inv_cfg,
)

print(out["wrms"])
```

### Solving for unknown horizontal and vertical reference shifts

If the horizontal or vertical velocity reference frame is uncertain, the inversion can solve for constant nuisance shifts:

```python
inv_cfg = gs2d.SlipInversionConfig(
    alpha=1.0,
    smoothing_order="second",
    solver_type="nonnegative",
    inversion_mode="forward_slip",
    use_vertical=True,
    solve_horizontal_shift=True,
    solve_vertical_shift=True,
)

out = gs2d.fit_profile_slip(
    observations,
    greens,
    inv_cfg,
)

print(out["horizontal_shift"])
print(out["vertical_shift"])
```

The total prediction includes the slip contribution plus the estimated reference shifts. The output also includes slip-only predictions for comparison.

### Inversion from lon/lat velocity files

GeoSlip2D also supports profile projection from lon/lat velocity data files.

Expected input columns are:

```text
longitude, latitude, Ve, Vn, Vu, sigma_e, sigma_n, sigma_u
```

A full workflow can be run with:

```python
out = gs2d.run_profile_slip_inversion(
    data_filename="examples/data/synthetic_data.txt",
    greens="Greens_homogeneous.npz",
)
```

---

## Example notebooks

The GitHub repository includes four primary example notebooks.

### `examples/Compliant_wedge_model.ipynb`

Demonstrates:

- building a shared interface geometry,
- configuring the compliant wedge backend,
- plotting the wedge model geometry,
- computing wedge Green's functions,
- comparing wedge and homogeneous surface velocities,
- running synthetic inversions with homogeneous and wedge Green's functions.

### `examples/Layered_model.ipynb`

Demonstrates:

- building layered elastic Green's functions,
- comparing layered and homogeneous Green's functions on the same interface,
- plotting horizontal and vertical surface velocities for uniform backslip above a specified locking depth,
- performing synthetic slip inversions using homogeneous and layered Green's functions.

### `examples/Slip_inversion.ipynb`

Demonstrates:

- loading synthetic lon/lat velocity data,
- building or loading homogeneous Green's functions,
- projecting data onto a 1D profile,
- inverting for slip or slip-deficit rate,
- optionally solving for horizontal and vertical velocity reference shifts,
- plotting data fits, residuals, and recovered slip.

### `examples/Viscoelastic_cycle.ipynb`

Demonstrates:

- building viscoelastic-cycle Green's functions,
- converting native VECycle output into `Greens2D`,
- comparing viscoelastic-cycle and homogeneous predictions,
- plotting interseismic horizontal and vertical surface velocities,
- performing synthetic inversions with homogeneous and viscoelastic-cycle Green's functions.

---

## Uniform backslip examples

Several example notebooks compute synthetic surface velocities for uniform locking or backslip above a specified locking depth.

A typical pattern is:

```python
locking_depth = 35.0
backslip_rate = 1.0

depth = greens.interface.centers[:, 1]
slip = np.zeros(greens.interface.n_patch)
slip[depth <= locking_depth] = backslip_rate

v_hor = greens.Ghor @ slip
v_vert = greens.Gvert @ slip
```

This makes it possible to compare the predicted surface velocity fields from different Green's-function backends using the same slip distribution.

---

## Common validation checks

For any Green's-function object, the following checks should pass:

```python
assert greens.Ghor.shape[0] == greens.xobs.size
assert greens.Gvert.shape == greens.Ghor.shape
assert greens.Ghor.shape[1] == greens.interface.n_patch
assert np.all(np.isfinite(greens.Ghor))
assert np.all(np.isfinite(greens.Gvert))
```

A useful diagnostic is:

```python
print(greens.summary())
```

---

## Sign conventions

GeoSlip2D uses a package-level sign convention in which the stored Green's matrices are intended to be used consistently with the profile slip inversion tools.

The inversion configuration controls whether the model is interpreted as forward slip or slip deficit:

```python
inv_cfg = gs2d.SlipInversionConfig(
    inversion_mode="forward_slip",
)
```

or:

```python
inv_cfg = gs2d.SlipInversionConfig(
    inversion_mode="slip_deficit",
)
```

When adding new Green's-function backends or comparing to external MATLAB scripts, always verify the sign of both horizontal and vertical predictions.

---

## Development notes

### Running tests

From the repository root:

```bash
pytest
```

### Editable install during development

```bash
pip install -e .
```

### Adding a new backend

New Green's-function backends should return a `Greens2D` object and should be wired through:

```python
gs2d.build_greens("backend_name", interface, xobs, config)
```

A backend should ensure:

- the returned `greens.xobs` matches the requested `xobs`,
- `greens.Ghor.shape[0] == len(xobs)`,
- `greens.Gvert.shape == greens.Ghor.shape`,
- `greens.Ghor.shape[1] == greens.interface.n_patch`,
- metadata clearly records the backend configuration and sign convention.

---

## Known limitations

- The homogeneous backend is the simplest and most transparent reference implementation.
- The layered backend is implemented but should be compared against trusted regression outputs when changing numerical details.
- The compliant wedge backend is fully vendored into GeoSlip2D but can be computationally expensive for fine discretizations.
- The viscoelastic-cycle backend is integrated into GeoSlip2D and returns `Greens2D`, but it currently uses its own native geometry internally before conversion.
- Large Green's-function calculations may take time, especially wedge and viscoelastic-cycle models.

---

## Citation and attribution

If you use GeoSlip2D in published work, cite the associated manuscript, software release, or repository DOI once available.

Some modules are Python ports or refactors of earlier MATLAB workflows for elastic dislocations, compliant wedge calculations, layered elastic calculations, and viscoelastic earthquake-cycle Green's functions.

---

## License

Add your preferred license here, for example:

```text
MIT License
```

or:

```text
BSD 3-Clause License
```

---

## Contact

Maintainer:

```text
Kaj Johnson
Indiana University Bloomington
Earth and Atmospheric Sciences
```
