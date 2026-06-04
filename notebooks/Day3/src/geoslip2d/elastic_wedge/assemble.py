from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np
from scipy import sparse

from .geometry import Geometry, SourceBlock
from .kernels import Kernels
from .params import ElasticWedgeParams


@dataclass
class System:
    blocks: Dict[str, np.ndarray]
    Gs11: np.ndarray
    Gs12: np.ndarray
    Gs21: np.ndarray
    Gs22: np.ndarray
    Gd11: np.ndarray
    Gd12: np.ndarray
    Gd21: np.ndarray
    Gd22: np.ndarray
    G: np.ndarray
    surfaceUxMat: np.ndarray
    surfaceUzMat: np.ndarray
    interfaceTauMat: np.ndarray
    sourceBlocks: list
    receiverSides: list
    mu: np.ndarray


def _assemble_disp_side(K, src_blocks, rec_boundary, rec_domain, component):
    parts = []
    for src in src_blocks:
        bsrc = src.boundary
        field = f"{component}_{src.side}"
        A = getattr(K[rec_boundary][bsrc], field)
        parts.append(A if src.domain == rec_domain else np.zeros_like(A))
    return np.hstack(parts)


def _assemble_stress_side(K, src_blocks, rec_boundary, rec_domain, field):
    parts = []
    for src in src_blocks:
        bsrc = src.boundary
        A = getattr(K[rec_boundary][bsrc], field)
        parts.append(A if src.domain == rec_domain else np.zeros_like(A))
    return np.hstack(parts)


def assemble_elastic_wedge_system(geom: Geometry, kernels: Kernels, params: ElasticWedgeParams) -> System:
    K = kernels.boundary
    src = geom.source_blocks
    rec = geom.receiver_sides
    mu = np.asarray(params.mu_vector, dtype=float)
    blocks = {}
    for r in rec:
        label = r.label
        b = r.boundary
        dom = r.domain
        blocks[f"Gd11_{label}"] = _assemble_disp_side(K, src, b, dom, "u11")
        blocks[f"Gd12_{label}"] = _assemble_disp_side(K, src, b, dom, "u12")
        blocks[f"Gd21_{label}"] = _assemble_disp_side(K, src, b, dom, "u21")
        blocks[f"Gd22_{label}"] = _assemble_disp_side(K, src, b, dom, "u22")
        blocks[f"Gs11_{label}"] = mu[dom] * _assemble_stress_side(K, src, b, dom, "sig11")
        blocks[f"Gs12_{label}"] = mu[dom] * _assemble_stress_side(K, src, b, dom, "sig12")
        blocks[f"Gs21_{label}"] = mu[dom] * _assemble_stress_side(K, src, b, dom, "sig21")
        blocks[f"Gs22_{label}"] = mu[dom] * _assemble_stress_side(K, src, b, dom, "sig22")

    vcat = np.vstack; hcat = np.hstack
    Gs11 = vcat([blocks["Gs11_1"], blocks["Gs11_2"], blocks["Gs11_3"],
                 blocks["Gs11_4t"]-blocks["Gs11_4b"], blocks["Gs11_5t"]-blocks["Gs11_5b"], blocks["Gs11_6t"]-blocks["Gs11_6b"]])
    Gs12 = vcat([blocks["Gs12_1"], blocks["Gs12_2"], blocks["Gs12_3"],
                 blocks["Gs12_4t"]-blocks["Gs12_4b"], blocks["Gs12_5t"]-blocks["Gs12_5b"], blocks["Gs12_6t"]-blocks["Gs12_6b"]])
    Gs21 = vcat([blocks["Gs21_1"], blocks["Gs21_2"], blocks["Gs21_3"],
                 blocks["Gs21_4t"]-blocks["Gs21_4b"], blocks["Gs21_5t"]-blocks["Gs21_5b"], blocks["Gs21_6t"]-blocks["Gs21_6b"]])
    Gs22 = vcat([blocks["Gs22_1"], blocks["Gs22_2"], blocks["Gs22_3"],
                 blocks["Gs22_4t"]-blocks["Gs22_4b"], blocks["Gs22_5t"]-blocks["Gs22_5b"], blocks["Gs22_6t"]-blocks["Gs22_6b"]])

    Gd11 = vcat([blocks["Gd11_4t"]-blocks["Gd11_4b"], blocks["Gd11_5t"]-blocks["Gd11_5b"], blocks["Gd11_6t"]-blocks["Gd11_6b"]])
    Gd12 = vcat([blocks["Gd12_4t"]-blocks["Gd12_4b"], blocks["Gd12_5t"]-blocks["Gd12_5b"], blocks["Gd12_6t"]-blocks["Gd12_6b"]])
    Gd21 = vcat([blocks["Gd21_4t"]-blocks["Gd21_4b"], blocks["Gd21_5t"]-blocks["Gd21_5b"], blocks["Gd21_6t"]-blocks["Gd21_6b"]])
    Gd22 = vcat([blocks["Gd22_4t"]-blocks["Gd22_4b"], blocks["Gd22_5t"]-blocks["Gd22_5b"], blocks["Gd22_6t"]-blocks["Gd22_6b"]])

    G = vcat([hcat([Gs11, Gs12]), hcat([Gs21, Gs22]), hcat([Gd11, Gd12]), hcat([Gd21, Gd22])])
    surfaceUxMat = vcat([hcat([blocks["Gd11_1"], blocks["Gd12_1"]]), hcat([blocks["Gd11_2"], blocks["Gd12_2"]]), hcat([blocks["Gd11_3"], blocks["Gd12_3"]])])
    surfaceUzMat = vcat([hcat([blocks["Gd21_1"], blocks["Gd22_1"]]), hcat([blocks["Gd21_2"], blocks["Gd22_2"]]), hcat([blocks["Gd21_3"], blocks["Gd22_3"]])])
    interfaceTauMat = vcat([hcat([blocks["Gs11_5t"], blocks["Gs12_5t"]]), hcat([blocks["Gs11_6t"], blocks["Gs12_6t"]])])

    if params.use_sparse:
        G = sparse.csc_matrix(G)
        surfaceUxMat = sparse.csc_matrix(surfaceUxMat)
        surfaceUzMat = sparse.csc_matrix(surfaceUzMat)
        interfaceTauMat = sparse.csc_matrix(interfaceTauMat)

    return System(blocks, Gs11, Gs12, Gs21, Gs22, Gd11, Gd12, Gd21, Gd22,
                  G, surfaceUxMat, surfaceUzMat, interfaceTauMat, src, rec, mu)
