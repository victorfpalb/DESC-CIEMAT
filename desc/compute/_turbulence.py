from desc.backend import jnp
from .data_index import register_compute_fun

@register_compute_fun(
name="f_turb",
label="f_{\\mathrm{turb}}",
units="~",
units_long="dimensionless",
description="Proxy for ITG turbulence, inspired by Landremann JPP 2025. 
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
    # - f_stab -

    # Bad curvature (K_y > 0) and good curvature regions (K_y < 0) regions, using DESC varaibles: 
    #    K_y = (B \times \nabla B ) \dot \nabla \alpha = -sign(psi_r) *B**3 * gbdrift (ensure good sign of toroidal flux)
    Ky = (-jnp.sign(data["psi_r"]) * data["|B|"] ** 3 * data["gbdrift"])    
    width = kwargs.get("smooth_width", 1e-3) # smooth transition for the Heaviside
    data["H_k_y"] = 0.5 * ( 1.0 + jnp.tanh(Ky / width)) # Heaviside function

    data["f_turb"] = (data["H_k_y"] + 0.4) * data["a"] * data["g^rr"]**(1/2) / jnp.sqrt(data["|B|"]) # f_stab as defined in doi:10.1017/S0022377825100536. 

    return data
