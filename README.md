

# README
- [Usage](#usage)
	* [Online calculations](#online-calculations)
	* [Post-processing](#post-processing)
- [Installation](#installation)
- [Implementation](#implementation)
	* [List of modified files](#list-of-modified-files)
- [Caveats/Limitations](#caveats-limitations)
	* [Changes to the dynamical core](#changes-to-the-dynamical-core)
- [Tests](#tests)
	* [Test details](#test-details)
	* [Installation](#installation-1)
- [Theory](#theory)
	* [Advection equation transformations](#advection-equation-transformations)
	* [Numerical implementation](#numerical-implementation)
	* [Alternative corrections](#alternative-corrections)
	* [Averaging and decomposition](#averaging-and-decomposition)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

This is a fork of the "Weather Research and Forecast model (WRF)" available at: [https://github.com/wrf-model/WRF](https://github.com/wrf-model/WRF).

**WRFlux** allows to output time-averaged resolved and subgrid-scale (SGS) fluxes and other tendency components for potential temperature, water vapor mixing ratio, and momentum for the ARW dynamical core. The included post-processing tool written in Python can be used to calculate tendencies from the fluxes, transform the tendencies to the Cartesian coordinate system, average spatially, and decompose the resolved advection into mean and resolved turbulent components. The sum of all forcing terms agrees to high precision with the model-computed tendency. The package is well tested and easy to install.
It is continuously updated when new WRF versions are released.
I'm currently preparing a publication that introduces WRFlux.

## Usage

### Online calculations
During the model run, fluxes, tendencies, and budget variables are averaged over time.
The online calculations can be controlled in the namelist file or the registry file `Registry/registry.wrflux`.

The following namelist variables are available:

- **`output_{t,q,u,v,w}_fluxes`** (default: 0): controls calculation and output for each variable;
 								  0: no output, 1: resolved fluxes + SGS fluxes + other source terms, 2: resolved fluxes only, 3: SGS fluxes only
- **`output_{t,q,u,v,w}_fluxes_add`** (default: 0): if 1, output additional fluxes using 2nd order advection and different correction forms for comparison (see [Theory/Alternative Corrections](#alternative-corrections)).
- **`output_dry_theta_fluxes`** (default: .true.): if .true., output fluxes based on dry theta even when the model uses moist theta (`use_theta_m=1`) internally.
- **`output_t_fluxes_small`** (default: 0): if 1, take into account theta-fluxes from acoustic time-step (slightly better budget closure unless `output_dry_theta_fluxes=.true.` and `use_theta_m=1`).
- **`hesselberg_avg`** (default: .true.): if .true., budget variables are averaged with density-weighting (see [Theory/Averaging and Decomposition](#averaging-and-decomposition))

SGS fluxes include horizontal and vertical fluxes from the diffusion module depending on `km_opt` and vertical fluxes from the boundary layer scheme.
The time-averaged fluxes are output in kinematic form (divided by mean dry air mass) and in the Cartesian coordinate system (see [Implementation](#implementation)).

The other source terms that are output beside resolved and SGS fluxes for `output_{t,q,u,v,w}_fluxes=1` are the following:

- **t**: microphysics, radiation (SW and LW)
- **q**: microphysics
- **u,v,w**: Coriolis and curvature, pressure gradient (from RK and acoustic step)
- **w**: buoyancy (from RK and acoustic step) and update of lower boundary condition (both share output variable with pressure gradient tendency)


All variables are output to the auxiliary output stream `auxhist24`. The **time-averaging interval** is set by the output interval `auxhist24_interval_m` and `auxhist24_interval_s`.

In addition to the namelist variables `output_{t,q,u,v,w}_fluxes` and `output_{t,q,u,v,w}_fluxes_add` you can of course control the output by changing the entries in `registry.wrflux` or using an iofields file. Instantaneous fluxes are by default not output.


### Post-processing

In the post-processing, the tendencies are calculated by differentiating the fluxes and decomposed into mean, resolved turbulent, and SGS. To check the closure of the budget, all forcing terms are added and the total model tendency over the averaging interval is calculated. The post-processing is done with a python package located in the directory `wrflux`. The tendency calculations can be done with the function `tools.calc_tendencies`. A template script is given by `tendency_calcs.py`. This script sets the arguments and runs `tools.calc_tendencies` for some example WRF output data located in the directory `example`. Then the output is checked for consistency and a profile plot is drawn.

The budget for each variable can be calculated in several different ways specified by the `budget_methods` argument. This is a list of strings, where each string is a combination of the following settings separated by a space:

- `cartesian`: advective tendencies in Cartesian instead of native form
- `dz_out_x`: use alternative corrections with derivatives of z taken out of temporal and horizontal derivatives;
horizontal corrections derived from horizontal flux (requires cartesian)
- `dz_out_z`: use alternative corrections with derivatives of z taken out of temporal and horizontal derivatives;
horizontal corrections derived from vertical flux (requires cartesian)
- `force_2nd_adv`: use 2nd-order advection

For the tendencies in Cartesian form, corrections are applied to the total tendency and to the horizontal derivatives and the vertical flux is using the Cartesian vertical velocity. See [Theory/Advection equation transformations](#advection-equation-transformations) for details. For an explanation of `dz_out_x` and `dz_out_z`, see [Theory/Alternative Corrections](#alternative-corrections).

If the WRF output data is too large to fit into memory, the domain can be decomposed into xy-tiles and processed tile per tile. The tile sizes are set in the `chunks` argument.
The tiles can also be processed in parallel:
```sh
 mpiexec -n N ./tendency_calcs.py
```

Other arguments of `tools.calc_tendencies` include: averaging directions (x and/or y) for horizontal average, time-averaging interval (if time-averaged data should be coarsened again before processing), and definition of a limited subset (in time and/or space) to process.



## Installation

The post-processing package in the directory `wrflux` can be installed together with its dependencies (xarray, matplotlib, netcdf4, and bottleneck) using `pip` (inside directory `wrflux`):

`pip install -e .`

To be able to do parallel processing with MPI an mpi-enabled version of `netcdf4-python` is required.
The easiest way to install this is using [`conda`](https://docs.conda.io/en/latest/miniconda.html) (inside directory `wrflux`):
```sh
conda create -n wrflux python=3.7
conda activate wrflux
conda install -c conda-forge netcdf4=*=mpi* xarray matplotlib bottleneck
pip install -e .
```
This also installs an MPI library. Note that when this conda environment is activated, the commands `mpiexec`, `mpif90`, etc. will point to the binaries in the conda environment. This can cause problems when compiling WRF.

Check if everything works as expected by running the example script:
```sh
python tendency_calcs.py
```
or 
```sh
mpiexec -n N ./tendency_calcs.py
```

## Implementation

The SGS fluxes are taken directly out of the diffusion routines in `module_diffusion_em.F`. For momentum fluxes, this is already implemented in the official version with the namelist variable `m_opt`. `m_opt` is automatically turned on when using WRFlux. 

The resolved fluxes are directly taken from the advection routines in `module_advect_em.F`, except for the vertical fluxes. 
The vertical fluxes are output in Cartesian form by multiplying the Cartesian vertical velocity with the correctly staggered budget variable. However, instead of using the vertical velocity calculated by WRF, it is recalculated based on the [equation given in Theory/Advection equation transformations](#w_eq).
This recalculated w is almost identical to the prognostic w. Only the last component in the equation is calculated in a slightly different way to be more consistent with the vertical advection in the native coordinate system (see also next section). The horizontal terms (terms 2 and 3 in the equation) are directly taken from the geopotential equation.

When decomposing the resolved flux into mean and resolved turbulent components in the post-processing (see [Theory/Averaging and Decomposition](#averaging-and-decomposition)), the turbulent component is calculated as a residual of the other two components. The same is done for the resolved advective tendency.

When spatial averaging is switched on in the post-processing, the mean flux does not contain the averaging dimension anymore and thus cannot be differentiated with respect to that dimension. Therefore, the advective tendency is attributed totally to the resolved turbulent component in that direction.

Map-scale factors are taken care of as described in WRF's [technical note](https://www2.mmm.ucar.edu/wrf/users/docs/technote/contents.html).
All output variables are decoupled from the map-scale factors.


### List of modified files
The following WRF source code files have been modified:

- Registry/Registry.EM_COMMON
- Registry/registry.em_shared_collection
- Registry/registry.les
- Registry/registry.wrflux
- dyn_em/module_advect_em.F
- dyn_em/module_avgflx_em.F
- dyn_em/module_big_step_utilities_em.F
- dyn_em/module_diffusion_em.F
- dyn_em/module_em.F
- dyn_em/module_first_rk_step_part1.F
- dyn_em/module_first_rk_step_part2.F
- dyn_em/module_initialize_ideal.F
- dyn_em/module_small_step_em.F
- dyn_em/solve_em.F
- dyn_em/start_em.F
- phys/module_pbl_driver.F
- share/module_check_a_mundo.F
- share/output_wrf.F
- wrftladj/solve_em_ad.F
- wrftladj/solve_em_tl.F




## Caveats/Limitations
Note the following limitations:

* For hydrostatic simulations (`non_hydrostatic=.false.`) the w-budget is not correct.
* SGS horizontal fluxes can only be retrieved for `diff_opt=2`.
* Tendencies from the cumulus scheme, sixth-order diffusion, and damping are not yet output explicitly. However, the residual of the budget can be used to assess them.
* For non-periodic boundary conditions, the budget calculation for the boundary grid points does not work.
* If using WENO or monotonic advection for scalars, energy is not perfectly conserved. Therefore, when the model uses moist theta (`use_theta_m=1`), the dry theta-budget obtained with `output_dry_theta_fluxes=.true.` is not closed. 

WRFlux has a relatively strong impact on the runtime of the model. If all budget variables are considered (`output_{t,q,u,v,w}_fluxes=1` for all variables, but `output_{t,q,u,v,w}_fluxes_add=0`), the runtime is increased by about 50 %.

### Changes to the dynamical core
The model evolution is almost identical to the current official version. However, two details in the dynamical core are different:

1. The vertical advection of geopotential (subroutine `rhs_ph` in `module_big_step_utilities_em.F`) is adapted to avoid double staggering of ![](https://latex.codecogs.com/svg.latex?\omega). This modification is currently under review and will probably be published in WRF's next major release (see [PR 1338](https://github.com/wrf-model/WRF/pull/1338/)). The modification makes the vertical advection of geopotential more consistent with the vertical advection of other variables and thereby reduces the differences between prognostic w and w diagnosed from the [equation](#w_eq) in the theory section to a minimum.
2. The conversion of the non-local flux from dry to moist theta for `km_opt=5` uses the moisture at each level instead of the surface moisture. In effect, the modification in [PR 1352](https://github.com/wrf-model/WRF/pull/1352) is not adopted, because it is not correct, in my opinion. My arguments for this standpoint are discussed in [PR 1259](https://github.com/wrf-model/WRF/pull/1259). The differences between the two approaches are, however, hardly noticeable in practice.




## Tests
### Test details
This package includes a test suite for automated testing with `pytest`. Idealized test simulations are run for one hour with a large number of different namelist settings to check all parts of the code including different combinations of `km_opt`, `bl_pbl_physics`, `isfflx`, `use_theta_m`, `output_dry_theta_fluxes`, `output_t_fluxes_small`, `hesselberg_avg`, `*adv_order`, `*adv_opt`, `mp_physics`, and boundary conditions. To check the output of WRFlux for consistency the following tests are carried out for these simulations:

- closure of budget (model tendency = summed forcing) in native and Cartesian grid (e > 0.9993)
- Cartesian = native advective tendencies if spatial directions are summed up (e > 0.99999)
- total advective tendency =  mean + resolved turbulent component (e > 0.9999999999)
- instantaneous vertical velocity very similar to the instantaneous [diagnosed vertical velocity](#w_eq) used in the tendency calculations (e > 0.995)
- no NaN and infinity values appearing except on the lateral boundaries for non-periodic BC
- 2nd-order tendencies equivalent to model tendencies if 2nd-order advection is used in the model (e > 0.999)
- similar advective tendencies for standard Cartesian corrections and [`dz_out_z` type corrections](#dz_out_z) (e > 0.9)

The last test is only done for one simulation.
All simulations, except for the ones with non-periodic BC, contain a 2D mountain ridge to turn on the effect of the Cartesian corrections.
Most simulations use random map-scale factors between 0.9 and 1 to make sure these are treated correctly.

The test statistic used is the Nash-Sutcliffe efficiency coefficient (NSE) calculated for data d with respect to the reference r:

![](https://latex.codecogs.com/svg.latex?e=1-\frac{\mathrm{MSE}(d,r)}{\mathrm{VAR}(r)}=1-\frac{\overline{(d-r)^2}}{\overline{(r-\bar{r})^2}})

The NSE is calculated over time, height, and along-mountain direction. Afterward, the minimum NSE value is taken over the remaining dimensions (cross-valley direction, and potentially flux direction, component (mean, resolved turbulent, or total), and budget method). The tests fail if the NSE is below the threshold given in brackets for the tests in the list. 

For some simulations, horizontal averaging in the along-mountain direction is tested (with a reduced threshold for the second test).

A shorter simulation is run with output at every time step from which turbulent fluxes are calculated explicitly. The thresholds for tests 2 and 3 are reduced in that case.

All test simulations are repeated with a short runtime (2 minutes) in debugging mode (WRF configured with `configure -D`) to detect floating-point exceptions and other issues.


### Installation
To run the test suite, `pytest` and my other WRF-related package [`run_wrf`](https://github.com/matzegoebel/run_WRF) are required. `run_wrf` facilitates the automatic setup and running of idealized WRF simulations (locally or using a batch system) given a configuration file that defines the simulations to run. It only runs on Linux. Download and install it with `pip`:


```sh
#install scipy with conda or later with pip
conda install -c conda-forge scipy
git clone https://github.com/matzegoebel/run_WRF.git
cd run_WRF
pip install -e .
```
Then go back to the directory `wrflux` and run:
`pip install -e .[test]`

To run the test suite you need to have two parallel builds of WRFlux, one with and one without the debugging option (compiled with `configure -D`). Specify the location of these builds in the configuration file `config_test_tendencies.py` in the variable `build_path`.
The names of the folders of the two builds are specified by the variables `parallel_build` and `debug_build` in the configuration file.

To run the test suite, execute `pytest` in the folder `wrflux/wrflux/test`. Make sure you do not have a python installation activated with an MPI library if you did not compile WRF with it. This would cause a problem when running the test simulations.
The test results are written to csv tables in the subdirectory `test_results`. For failed tests scatter plots are created in the subdirectory `figures`.


## Theory

### Advection equation transformations
The advection equation for a variable ![](https://latex.codecogs.com/svg.latex?\psi) in the Cartesian coordinate system in flux-form reads:

![](https://latex.codecogs.com/svg.latex?\partial_t\left({\rho}\psi\right)=\sum_{i=1}^{3}-\partial_{x_i}\left({\rho}u_i\psi\right))

Like many other atmospheric models, WRF uses a coordinate transformation from the Cartesian coordinate system ![](https://latex.codecogs.com/svg.latex?(t,x,y,z)) to ![](https://latex.codecogs.com/svg.latex?(t,x,y,\eta)) with the generalized vertical coordinate ![](https://latex.codecogs.com/svg.latex?\eta=\eta(x,y,z,t)). In WRF ![](https://latex.codecogs.com/svg.latex?\eta) is a hybrid terrain-following coordinate.

The transformed advection equation reads:

<a name="trans">

![](https://latex.codecogs.com/svg.latex?\partial_t\left({\rho}z_\eta\psi\right)=\sum_{i=1}^{2}\left[-\partial_{x_i}\left({\rho}z_{\eta}u_i\psi\right)\right]-\partial_{\eta}\left({\rho}z_\eta\omega\psi\right))

</a>

with the contra-variant vertical velocity ![](https://latex.codecogs.com/svg.latex?\omega) and the vertical coordinate metric ![](https://latex.codecogs.com/svg.latex?z_\eta).
In this and the following equations, all horizontal and temporal derivatives are taken on constant ![](https://latex.codecogs.com/svg.latex?\eta)-levels.
Note that in WRF the coordinate metric appears as part of the dry air mass ![](https://latex.codecogs.com/svg.latex?\mu_{\mathrm{d}}=-\rho_{\mathrm{d}}gz_\eta) in the equations.

In the following, I will derive a form of the advection equation in which the individual parts are the same as in the Cartesian advection equation for improved interpretability, but nevertheless, horizontal derivatives are taken on constant ![](https://latex.codecogs.com/svg.latex?\eta)-levels for ease of computation.

The relationship between the contra-variant vertical velocity ![](https://latex.codecogs.com/svg.latex?\omega) in the ![](https://latex.codecogs.com/svg.latex?\eta)-coordinate system and the Cartesian vertical velocity ![](https://latex.codecogs.com/svg.latex?w) is given by the geopotential equation:

<a name="w_eq">

![](https://latex.codecogs.com/svg.latex?w=\frac{\mathrm{d}z}{\mathrm{d}t}=(\partial_{t}\phi+u\partial_{x}\phi+v\partial_{y}\phi+\omega\partial_{\eta}\phi)/g=z_t+z_xu+z_yv+z_\eta\omega)

</a>

Solving for ![](https://latex.codecogs.com/svg.latex?z_\eta\omega), inserting in the previous equation, and rearranging leads to:

![](https://latex.codecogs.com/svg.latex?\partial_t\left({\rho}z_\eta\psi\right)-\partial_{\eta}\left({\rho}z_t\psi\right)=\sum_{i=1}^{2}\left[-\partial_{x_i}\left({\rho}z_{\eta}u_i\psi\right)+\partial_{\eta}\left({\rho}z_{x_i}u_i\psi\right)\right]-\partial_{\eta}\left({\rho}w\psi\right))

Dividing by ![](https://latex.codecogs.com/svg.latex?z_\eta) yields the back-transformed advection equation:

<a name="backtrans">

![](https://latex.codecogs.com/svg.latex?z_\eta^{-1}\partial_{t}\left({\rho}z_\eta\psi\right)-\partial_{z}\left({\rho}z_t\psi\right)=\sum_{i=1}^{2}\left[-z_\eta^{-1}\partial_{x_i}\left({\rho}z_{\eta}u_i\psi\right)+\partial_{z}\left({\rho}z_{x_i}u_i\psi\right)\right]-\partial_{z}\left({\rho}w\psi\right))

</a>

Using the product rule and the commutativity of partial derivatives we can show that this equation is analytically equivalent to the following:

<a name="dz_out">

![](https://latex.codecogs.com/svg.latex?\partial_{t}\left({\rho}\psi\right)-\partial_{z}\left({\rho}\psi\right)z_t=\sum_{i=1}^{2}\left[-\partial_{x_i}\left({\rho}u_i\psi\right)+\partial_{z}\left({\rho}u_i\psi\right)z_{x_i}\right]-\partial_{z}\left({\rho}w\psi\right))

</a>

This equation follows straight from the Cartesian advection equation. The advective tendency (left-hand side) and the horizontal advection components have a correction term added that accounts for the derivatives being taken on constant ![](https://latex.codecogs.com/svg.latex?\eta) instead of constant height levels.
Numerically however, this form is not perfectly consistent with the [original transformed advection equation](#trans) (see also [Alternative Corrections](#alternative-corrections)).

Thus, we stay with the previous [back-transformed equation](#backtrans), in which the individual components plus their corrections are equivalent to the components of the Cartesian advection equation. 

### Numerical implementation

WRF uses a staggered grid, where the fluxes of ![](https://latex.codecogs.com/svg.latex?\psi)  are staggered with respect to ![](https://latex.codecogs.com/svg.latex?\psi). If ![](https://latex.codecogs.com/svg.latex?\psi) is on the mass grid (potential temperature and mixing ratio) the equation with staggering operations indicated reads:

![](https://latex.codecogs.com/svg.latex?z_\eta^{-1}\partial_{t}\left({\rho}z_\eta\psi\right)-\partial_{z}\left({\rho}z_t\overline{\psi}^z\right)=\sum_{i=1}^{2}\left[-z_\eta^{-1}\partial_{x_i}\left({\rho}z_{\eta}u_i\overline{\psi}^{x_i}\right)+\partial_{z}\left({\rho}z_{x_i}\overline{u_i}^{x_iz}\overline{\psi}^z\right)\right]-\partial_{z}\left({\rho}w\overline{\psi}^z\right))

where the staggering operations are denoted with an overbar and the staggering direction. For momentum, the equation looks a bit differently, since also the velocities in the fluxes need to be staggered. The staggering of ![](https://latex.codecogs.com/svg.latex?\psi) depends on the advection order as described in WRF's [technical note](https://www2.mmm.ucar.edu/wrf/users/docs/technote/contents.html). For the [back-transformed advection equation](#backtrans) to be numerically consistent with the [original transformed advection equation](#trans), all derivatives need to use the correct advection order. The correction terms derive from the vertical advection term and thus must use the order of the vertical advection.


### Alternative corrections

Before, we introduced a [form of the advection equation](#dz_out) in which the derivatives of z were taken out of the temporal and horizontal derivatives.
For comparison, this form is also implemented in the package in two different ways:
The straightforward implementation takes the horizontal flux and staggers it horizontally and vertically to the grid of the vertical flux (budget setting `dz_out_x`):

![](https://latex.codecogs.com/svg.latex?\partial_{t}\left({\rho}\psi\right)-\partial_{z}\left({\rho}\overline{\psi}^z\right)z_t=\sum_{i=1}^{2}\left[-\partial_{x_i}\left({\rho}u_i\overline{\psi}^{x_i}\right)+\partial_{z}\left(\overline{{\rho}u_i\overline{\psi}^{x_i}}^{x_iz}\right)\overline{z_{x_i}}^{x_i}\right]-\partial_{z}\left({\rho}w\overline{\psi}^{z}\right))

This implementation is analogous to how SGS fluxes are corrected in WRF.

The second implementation is a hybrid form in which the correctly staggered ![](https://latex.codecogs.com/svg.latex?\overline{\psi}^{z}) is used (budget setting `dz_out_z`):

<a name="dz_out_z">

![](https://latex.codecogs.com/svg.latex?\partial_{t}\left({\rho}\psi\right)-\partial_{z}\left({\rho}\overline{\psi}^z\right)z_t=\sum_{i=1}^{2}\left[-\partial_{x_i}\left({\rho}u_i\overline{\psi}^{x_i}\right)+\partial_{z}\left({\rho}\overline{u_i}^{x_iz}\overline{\psi}^{z}\right)\overline{z_{x_i}}^{x_i}\right]-\partial_{z}\left({\rho}w\overline{\psi}^{z}\right))

</a>

This implementation leads to a much better closure of the budget as the previous one, but not quite as good as the [back-transformed advection equation](#backtrans).


### Averaging and decomposition
When the advection equation is averaged over time and/or space, the fluxes and resulting tendency components can be decomposed into mean advective and resolved turbulent components.
Since WRF is a compressible model, we use a density-weighted average (Hesselberg averaging) unless `hesselberg_avg=.false.`. The effect of the density-weighting is rather small. 

Means and perturbations are defined by:

![](https://latex.codecogs.com/svg.latex?\widetilde{\psi}=\frac{\left\langle\rho\psi\right\rangle}{\left\langle\rho\right\rangle},\quad\psi'':=\psi-\widetilde{\psi})

![](https://latex.codecogs.com/svg.latex?\langle\psi\rangle) denotes the time and/or spatial average, ![](https://latex.codecogs.com/svg.latex?\widetilde{\psi}) is the density-weighted average, and ![](https://latex.codecogs.com/svg.latex?\psi'') the perturbation.

The flux decomposition then reads:

![](https://latex.codecogs.com/svg.latex?\left\langle{\rho}u_i\psi\right\rangle=\left\langle\rho\right\rangle\widetilde{u_i}\widetilde{\psi}+\left\langle{\rho}u_i''\psi''\right\rangle) for ![](https://latex.codecogs.com/svg.latex?i={1,2,3}).

The correction flux is decomposed like this:

![](https://latex.codecogs.com/svg.latex?\left\langle{\rho}Z_i\psi\right\rangle=\left\langle\rho\right\rangle\widetilde{Z_i}\widetilde{\psi}+\left\langle{\rho}Z_i''\psi''\right\rangle) for ![](https://latex.codecogs.com/svg.latex?i={1,2}) 

with ![](https://latex.codecogs.com/svg.latex?Z_i=z_{x_i}u_i).

Note that the time average is a block average, not a running average.


## Contributing

Feel free to report [issues](https://github.com/matzegoebel/WRFlux/issues) on Github.
You are also invited to fix bugs and improve the code yourself. Your changes can be integrated with a [pull request](https://github.com/matzegoebel/WRFlux/pulls).

## Acknowledgments

Thanks to Lukas Umek who provided the code used as starting point for WRFlux.


