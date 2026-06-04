"""Input/output helpers for GeoSlip2D Green's structures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

from .geometry import interface_from_arrays
from .greens import Greens2D, greens_from_matdict


def save_greens(filename: str | Path, greens: Greens2D, *, mat_struct_name: str = "Greens") -> None:
    """Save a :class:`Greens2D` object.

    ``.npz`` is the native GeoSlip2D format.  ``.mat`` is supported for MATLAB
    compatibility and writes a MATLAB-style struct named ``mat_struct_name``.
    """
    filename = Path(filename)
    suffix = filename.suffix.lower()
    if suffix == ".npz":
        _save_greens_npz(filename, greens)
    elif suffix == ".mat":
        savemat(filename, {mat_struct_name: greens.to_matdict()})
    else:
        raise ValueError("save_greens supports only .npz and .mat files.")


def load_greens(filename: str | Path, *, mat_struct_name: str | None = None) -> Greens2D:
    """Load a Green's-function object from ``.npz`` or ``.mat``."""
    filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError(f"Green's-function file does not exist: {filename}")
    suffix = filename.suffix.lower()
    if suffix == ".npz":
        return _load_greens_npz(filename)
    if suffix == ".mat":
        mat = loadmat(filename, squeeze_me=True, struct_as_record=False)
        user_vars = [k for k in mat if not k.startswith("__")]
        if mat_struct_name is None:
            if len(user_vars) != 1:
                raise KeyError(
                    "MAT-file contains multiple variables; pass mat_struct_name. "
                    f"Available variables: {', '.join(user_vars)}"
                )
            mat_struct_name = user_vars[0]
        if mat_struct_name not in mat:
            raise KeyError(f"'{mat_struct_name}' not found. Available variables: {', '.join(user_vars)}")
        return greens_from_matdict(_mat_struct_to_dict(mat[mat_struct_name]), source_type="mat")
    raise ValueError("load_greens supports only .npz and .mat files.")


def _json_ready(obj: Any) -> Any:
    """Convert NumPy-rich metadata into JSON-serializable Python objects.

    Green's-function metadata often contains arrays, NumPy scalar types, or
    nested dataclasses/dicts.  The native ``.npz`` writer stores metadata as a
    JSON string, so these values need to be converted before ``json.dumps``.
    Complex arrays are preserved as dictionaries with real/imag parts.
    """
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, np.generic):
        return _json_ready(obj.item())
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj)
        if np.iscomplexobj(arr):
            return {
                "__complex_array__": True,
                "real": np.real(arr).tolist(),
                "imag": np.imag(arr).tolist(),
            }
        return arr.tolist()
    if isinstance(obj, complex):
        return {"__complex__": True, "real": obj.real, "imag": obj.imag}
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return _json_ready(vars(obj))
    return str(obj)


def _save_greens_npz(filename: Path, greens: Greens2D) -> None:
    np.savez(
        filename,
        Ghor=greens.Ghor,
        Gvert=np.empty((0, 0)) if greens.Gvert is None else greens.Gvert,
        has_Gvert=np.array(greens.Gvert is not None),
        xobs=greens.xobs,
        topx_interface=greens.interface.topx,
        topz_interface=greens.interface.topz,
        botx_interface=greens.interface.botx,
        botz_interface=greens.interface.botz,
        centers_interface=greens.interface.centers,
        patch_length_interface=greens.interface.patch_length,
        dip_interface=greens.interface.dip,
        source_type=np.array(greens.source_type),
        units=np.array(greens.units),
        sign_convention=np.array(greens.sign_convention),
        metadata_json=np.array(json.dumps(_json_ready(greens.metadata))),
        interface_metadata_json=np.array(json.dumps(_json_ready(greens.interface.metadata))),
    )


def _load_greens_npz(filename: Path) -> Greens2D:
    with np.load(filename, allow_pickle=False) as z:
        has_gvert = bool(np.asarray(z["has_Gvert"]).item()) if "has_Gvert" in z.files else "Gvert" in z.files
        interface = interface_from_arrays(
            topx=z["topx_interface"],
            topz=z["topz_interface"],
            botx=z["botx_interface"],
            botz=z["botz_interface"],
            centers=z["centers_interface"] if "centers_interface" in z.files else None,
            patch_length=z["patch_length_interface"] if "patch_length_interface" in z.files else None,
            dip=z["dip_interface"] if "dip_interface" in z.files else None,
            metadata=_loads_json_scalar(z, "interface_metadata_json"),
        )
        return Greens2D(
            Ghor=z["Ghor"],
            Gvert=z["Gvert"] if has_gvert else None,
            xobs=z["xobs"],
            interface=interface,
            source_type=_loads_str_scalar(z, "source_type", "unknown"),
            units=_loads_str_scalar(z, "units", "displacement_per_unit_slip"),
            sign_convention=_loads_str_scalar(z, "sign_convention", "forward_slip_positive"),
            metadata=_loads_json_scalar(z, "metadata_json"),
        )


def _loads_str_scalar(z: Any, key: str, default: str) -> str:
    if key not in z.files:
        return default
    return str(np.asarray(z[key]).item())


def _loads_json_scalar(z: Any, key: str) -> dict[str, Any]:
    if key not in z.files:
        return {}
    text = str(np.asarray(z[key]).item())
    return json.loads(text) if text else {}


def _mat_struct_to_dict(obj: Any) -> dict[str, Any]:
    """Convert scipy-loaded MATLAB structs to plain dictionaries."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "_fieldnames"):
        return {name: getattr(obj, name) for name in obj._fieldnames}
    if isinstance(obj, np.ndarray) and obj.dtype.names is not None:
        squeezed = np.squeeze(obj)
        return {
            name: squeezed[name].item() if np.asarray(squeezed[name]).size == 1 else squeezed[name]
            for name in obj.dtype.names
        }
    raise TypeError("Could not convert MATLAB object to dictionary.")
