from desc.backend import jnp
from .data_index import register_compute_fun


@register_compute_fun(
    name="f_ITG",
    label="f_{\\mathrm{ITG}}",
    units="~",
    units_long="dimensionless",
    description="Local ITG turbulence proxy",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    data=["psi_r", "gbdrift", "g^rr", "|B|", "a"],
    coordinates="rtz",
)
def _f_ITG(params, transforms, profiles, data, **kwargs):
    """Local ITG turbulence proxy.

    Smooth version for optimization:

        f_itg = (H_smooth(Ky) + 0.4) * |∇rho| / sqrt(|B|)

    with:

        Ky = - sign(psi_r) * B**3 * gbdrift 

    and:

        H_smooth(Ky) = 0.5 * (1 + tanh(Ky / width))

    For width -> 0, this tends to the hard Heaviside.
    """

    Ky = - jnp.sign(data["psi_r"]) * (data["|B|"]**3) * data["gbdrift"] # Prueba con el B**3!
    data["K_y"] = Ky

    # Definition of a 'smooth' heaviside with a tanh. 
    smooth_width = kwargs.get("smooth_width", 1e-3)

    H = 0.5 * (1.0 + jnp.tanh( Ky / jnp.maximum(smooth_width, 1e-3)))
    data["H"] = H

    # NEW!
    # f_ITG = Heaviside(K_y) * |∇rho| / sqrt(|B|)
    data["f_ITG"] = H * jnp.sqrt(data["g^rr"]) / jnp.sqrt(data["|B|"])

    return data


@register_compute_fun(
    name="H_k_y",
    label="H_\\epsilon(K_y)",
    units="~",
    units_long="dimensionless",
    description="Smooth indicator of bad curvature (K_y > 0)",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    data=["psi_r", "gbdrift", "|B|"],
    coordinates="rtz",
    smooth_width="float: Positive smoothing width in K_y units. Default 1e-3.",
)
def _H_bad_curvature(params, transforms, profiles, data, **kwargs):
    """Smooth indicator of bad curvature."""
    Ky = (-jnp.sign(data["psi_r"]) * data["|B|"] ** 3 * data["gbdrift"])

    width = kwargs.get("smooth_width", 1e-3)

    data["H_k_y"] = 0.5 * ( 1.0 + jnp.tanh(Ky / width))

    return data

@register_compute_fun(
name="f_turb",
label="f_{\\mathrm{turb}}",
units="~",
units_long="dimensionless",
description="Proxy for ITG turbulence, inspired by T. Schuett talk in TSVV Metting 2026",
dim=1,
params=[],
transforms={},
profiles=[],
data=["psi_r", "|B|", "cvdrift", "a", "g^rr", "gbdrift"],
coordinates="rtz",
smooth_width="float: Positive smoothing width in K_y units. Default 1e-3.",
)
def _f_turb(params, transforms, profiles, data, **kwargs):
    """
    Proxy inspired in Landreman JPP 2025, for ITG optimization
    """
    # Bad curvature (K_y > 0) and good curvature regions (K_y < 0) regions, using DESC varaibles: 
    #    K_y = (B \times \kappa ) \dot \nabla \alpha = B**2 * cvdrift * -sign(psi_r) (ensure good sign of toroidal flux)
    
    Ky = (-jnp.sign(data["psi_r"]) * data["|B|"] ** 2 * data["cvdrift"])

    # Definition of a smooth Heaviside function of K_y. 
    width = kwargs.get("smooth_width", 1e-3)
    data["H_k_y"] = 0.5 * ( 1.0 + jnp.tanh(Ky / width))

    # Proxy as defined by T. Schuett, but transformed into DESC coordinates:
    #
    #   X(s) = mean[ H( B \times \kappa \dot \nabla\alpha) ( a \nabla s)**2  ]
    # 
    #   X(rho) ≃ mean[ H( B**2 * cvdrift ) a**2 g^rr ]

    data["f_turb"] = data["H_k_y"] * data["a"]**2 * data["g^rr"] 

    ### NEW! Landremann description

    # - f_Q -
    # data["f_turb"] = (data["H_k_y"] + 0.2) * data["a"]**3 * data["g^rr"]**(3/2) / data["|B|"] # f_Q

    # - f_stab -
    # Ky = (-jnp.sign(data["psi_r"]) * data["|B|"] ** 3 * data["gbdrift"])    
    # width = kwargs.get("smooth_width", 1e-3)
    # data["H_k_y"] = 0.5 * ( 1.0 + jnp.tanh(Ky / width))
    # data["f_turb"] = (data["H_k_y"] + 0.4) * data["a"] * data["g^rr"]**(1/2) / jnp.sqrt(data["|B|"]) # f_stab

    return data
