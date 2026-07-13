from desc.backend import jnp
from .data_index import register_compute_fun


@register_compute_fun(
    name="f_ITG",
    label="f_{\\mathrm{stab}}",
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
def _f_stab(params, transforms, profiles, data, **kwargs):
    """Local ITG turbulence proxy.

    Smooth version for optimization:

        f_itg = (H_smooth(Ky) + 0.4) * |∇rho| / sqrt(|B|)

    with:

        Ky = - sign(psi_r) * (a/2) * gbdrift # I should be B**3 * gbdrift according to DESC definition, but because
                                             # we just care of the magnitude, the sign for the Heaviside, this is fine. 

    and:

        H_smooth(Ky) = 0.5 * (1 + tanh(Ky / width))

    For width -> 0, this tends to the hard Heaviside.
    """

    Ky = - jnp.sign(data["psi_r"]) * (data["a"]/2) * data["gbdrift"]

    data["K_y"] = Ky

    smooth_width = kwargs.get("smooth_width", 1e-3)

    H = 0.5 * (
        1.0 + jnp.tanh( Ky / jnp.maximum(smooth_width, 1e-12))
    )

    data["f_ITG"] = ((H + 0.4) * jnp.sqrt(data["g^rr"]) / jnp.sqrt(data["|B|"]))

    return data
