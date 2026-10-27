from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid, Grid, QuadratureGrid
from desc.utils import Timer

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs

from desc.integrals import surface_averages

class F_turb(_Objective):
    r"""Proxy to reduce ITG turbulence. Based in Landreman JPP 2025 doi:10.1017/S0022377825100536. 
    
    f_stab = [ H (B \times \nabla B \dot \alpha) (a |\nabla \rho| )**2 / sqrt(|B|) ]

    where H is the  heaviside funtion (1 for regions of bad curvature, 0 otherwise). 

    The proxy is calculated along a given field line α, in a flux surface ρ. Considering npol poloidal (θ) turns, 
    this implies exploring along ζ: ζ = θ / ι(ρ) , where ι(ρ) is the rotational transform in the specified surface. 

    Two residuals will be considered:
        - If mean=True --> r_i = max ( mean(X(ρ)) - mean (X*), 0). 
          In this case, the max() will be taken between two scalars. 
          The size of the residual will be the # of surfaces beign considered.

        - If mean=False --> r_i = max ( X(ρ) - X*, 0)
          In this case, the max will be taken for each point along the field line, optimizing only those for which X > X*.
          The size of the residual will be the # of surfaces times the number of points along the field line. 

    NOTE: X* should be specified as an input of the function, as it would be the target X value to obtain. By default, we can set it to be a vector of 0
          (with the correct dimention according to the resiudal considered). In practice, it should be some percentage of the initial X of the equilibrium, and should be fixed 
          in the optimization script. E.g., X* = 0.7 * X_initial. 
    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
    )

    _equilibrium = True
    _coordinates = "rtz"
    _units = "~"
    _print_value_fmt = "f_turb: "

    _static_attrs = list(_Objective._static_attrs) + [
        "_grid",
        "_mean",
        "_surface_keys",
    ]

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
        name="f_turb",
        jac_chunk_size=None,
        mean=True, # choose type of residual 
        X_star=None, # choose target value
        rho=(0.3, 0.5, 0.7), # Default surfaces
        alpha=0.0, 
        npol=1,
        npts=128, # Resolution of the field line
    ):
        if target is None and bounds is None:
            target = 0

        # Surface integration grid, not the field-line grid.
        self._grid = grid

        self._mean = mean
        self._X_star = X_star
        self._rho = rho
        self._alpha = alpha
        self._npol = npol
        self._npts = npts

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

        # Grid to compute iota. It should be defined in a flux surface. 
        if self._grid is None:
            self._grid = LinearGrid(rho=jnp.atleast_1d(self._rho), M=4 * eq.M, N=eq.N, NFP=eq.NFP, sym=False,)

        grid_surface = self._grid
        rho = jnp.asarray(grid_surface.nodes)[grid_surface.unique_rho_idx, 0]

        nrho = len(rho)
        npts = self._npts

        # Volume grid needed to evalueate the minor radius of the configuration. 
        grid_volume = QuadratureGrid( L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP,)

        # Fixed sampling in the magnetic poloidal angle, taking into account the equilibrium iota.
        theta = self._alpha + jnp.linspace( -jnp.pi * self._npol,  jnp.pi * self._npol, npts, endpoint=False,)

        # For each poloidal sample, iterate over all surfaces.
        # This matches reshape((nrho, npts), order="F").
        rho_nodes = jnp.tile(jnp.asarray(rho), npts)
        theta_nodes = jnp.repeat(theta, nrho)
        rho_idx = jnp.tile(jnp.arange(nrho), npts)

        self._coordinates = "r" if self._mean else "rtz"
        self._dim_f = nrho if self._mean else nrho * npts # dimention of the residual depending on mean=True/False
        self._data_keys = ["f_turb"]
        self._surface_keys = ("iota", "iota_r", "iota_rr")

        # Fixed threshold. Average node-wise values if mean=True.
        X_star = jnp.asarray(  0.0 if self._X_star is None else self._X_star) # Set to zero if X* is not given. 

        if self._mean and X_star.shape == (nrho * npts,): 
            #   Perform the mean, if mean=True, but X* given is not averaged. 
            X_star = X_star.reshape((nrho, npts), order="F").mean(axis=1)

        # good shape for X*
        if not self._mean and X_star.shape == (nrho,):
            X_star = jnp.tile(X_star, npts)
        X_star = jnp.broadcast_to(X_star, (self._dim_f,)) # Ensure good dimention of X*

        # Only transforms on the fixed grids are precomputed!
        self._constants = {
            "grid_surface": grid_surface,
            "surface_transforms": get_transforms(self._surface_keys, obj=eq, grid=grid_surface),
            "surface_profiles": get_profiles(self._surface_keys, obj=eq, grid=grid_surface),
            "volume_transforms": get_transforms(["a"], obj=eq, grid=grid_volume),
            "volume_profiles": get_profiles(["a"], obj=eq, grid=grid_volume),
            "line_profiles": get_profiles(self._data_keys, obj=eq),
            "rho_nodes": rho_nodes,
            "theta": theta_nodes,
            "rho_idx": rho_idx,
            "unique_rho_idx": jnp.arange(nrho),
            "alpha": jnp.asarray(self._alpha),
            "X_star": X_star,
            "quad_weights": jnp.ones(self._dim_f),
        }

        # TODO: How to normalize this? Just for the output!
        self._normalization = 1.0

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Evaluate the current field line and its excess over X_star."""
        if constants is None:
            constants = self._constants

        eq = self.things[0]
        c = constants
        grid_surface = c["grid_surface"]
        nrho = grid_surface.num_rho

        # compute iota and related properties
        surface_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._surface_keys,
            params=params,
            transforms=c["surface_transforms"],
            profiles=c["surface_profiles"],
        )

        radial_data = {
            key: grid_surface.compress(surface_data[key])
            for key in self._surface_keys
        }

        # Expand surface values to match the field-line node ordering. This needed to use the 
        # exact iota for each field line. 
        data = {key: value[c["rho_idx"]] for key, value in radial_data.items()}

        # Update zeta to follow a field line at constant alpha.
        theta = c["theta"]
        zeta = (theta - c["alpha"]) / data["iota"] # α=0 = θ - ι(ρ) * ζ --> ζ = θ / ι(ρ)

        # Coordinates of the field line
        coords = jnp.column_stack([c["rho_nodes"], theta, zeta,])

        # Convert magnetic coordinates to DESC computational coordinates. In our description, we have used the assumtion that 
        # field lines are straight using α = θ - ι(ρ) * ζ, but this is not real in the computational domain of DESC. 
        # So we map the coordinates we have in magnetic coordinates (PEST) --> DESC computational coordinates. 

        nodes = eq.map_coordinates(
            coords,
            inbasis=("rho", "theta_PEST", "zeta"),
            outbasis=("rho", "theta", "zeta"),
            params=params,
            guess=coords,
            tol=1e-10,
        )

        # Construct the updated grid without reordering its nodes.
        grid_line = Grid(
            nodes,
            coordinates="rtz",
            NFP=eq.NFP,
            period=(jnp.inf, jnp.inf, jnp.inf), # The field line is not periodic! (unless rational surface).
            sort=False,
            jitable=True,
            _unique_rho_idx=c["unique_rho_idx"],
            _inverse_rho_idx=c["rho_idx"],
        )

        # Rebuild transforms because the field-line nodes have changed.
        transforms = get_transforms(
            self._data_keys, obj=eq, grid=grid_line, jitable=True,)

        # Compute the current minor radius 'a' on the volume grid.
        volume_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            ["a"],
            params=params,
            transforms=c["volume_transforms"],
            profiles=c["volume_profiles"],
        )
        data["a"] = volume_data["a"] # we define "a" in data, so it can be read by the compute function

        # 5. Evaluate the proxy along the updated field line.
        # Supplying iota and a avoids integration on the field-line grid.
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=transforms,
            profiles=c["line_profiles"],
            data=data,
        )

        X = data["f_turb"]

        if self._mean:
            X = X.reshape((nrho, -1), order="F").mean(axis=1)

        return jnp.maximum(X - c["X_star"], 0.0)