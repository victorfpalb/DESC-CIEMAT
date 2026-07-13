from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import Timer

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs

from desc.integrals import surface_averages

class ITG_turbulence(_Objective):
    r"""Proxy to reduce ITG turbulence. Based on eq. (5.2) of Landreman et al (2025), JPP doi:10.1017/S0022377825100536. 
    
    f_stab = mean( [ H(gbdrift) + 0.4 ] |nabla rho| / sqrt(B))

    where H is the  heaviside funtion (1 for regions of bad curvature, 0 otherwise). 

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
    )

    _equilibrium = True
    _coordinates = "rtz"
    _units = "~"
    _print_value_fmt = "f_stab: "

    _static_attrs = tuple(_Objective._static_attrs) + ("_grid", "_average")
    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="f_stab",
        jac_chunk_size=None,
        average=True,
    ):
        if target is None and bounds is None:
            target = 0

        self._grid = grid
        self._average = average

        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
        )

    def build(self, use_jit=True, verbose=1):
        eq = self.things[0]

        if self._grid is None:
            grid = LinearGrid(
                rho=jnp.sqrt(jnp.array([0.5, 0.7])),
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )
        else:
            grid = self._grid

        if self._average:
            self._coordinates = "r"
            self._dim_f = grid.num_rho
            self._data_keys = [
                "f_ITG",
                "sqrt(g)",
            ]

        else:
            self._coordinates = "rtz"
            self._dim_f = grid.num_nodes
            self._data_keys = [
                "f_ITG",
            ]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)

        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
            "grid": grid,
            "quad_weights": jnp.ones((self._dim_f,)), # same weight to each node
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            self._normalization = 1.0 # TODO: How we normalize this?

        super().build(use_jit=use_jit, verbose=verbose)


    def compute(self, params, constants=None):
        """Compute quantity to be minimized."""

        if constants is None:
            constants = self._constants

        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )

        if self._average:
            residual = surface_averages(
                constants["grid"],
                data["f_ITG"],
                sqrt_g=data["sqrt(g)"],
                surface_label="rho",
                expand_out=False,
            )
        else:
            residual = data["f_ITG"]

        return residual
