Solve and simulate the navier stokes equations for a time dependent rotating cylinder in a viscous fluid. The cylinder is made of a porous material with a pressure gradient pulling the fluid inside the cylinder. See the example below.

The radial fluid velocity can be solved analytically

$$u_r(r, t) = -\frac{U R}{r}$$

where R is radius of the cylinder and U is the radial velocity at $r=R$ (caused by the pressure).

The azimuthal velocity component needs to be solved numerically from the azimuthal Navier Stokes equation:

$$ \frac{\partial u_{\theta}}{\partial t} - \frac{U R}{r} \frac{\partial u_{\theta}}{\partial r} - \frac{U R}{r^2} u_{\theta} = \nu \left( \frac{\partial^2 u_{\theta}}{\partial r^2} + \frac{1}{r} \frac{\partial u_{\theta}}{\partial r} - \frac{u_{\theta}}{r^2} \right)$$

With boundary and initial conditions
$$u_{\theta}(R, t) = \Omega(t)R, \quad u_{\theta}(\infty, t) = 0, \quad u_{\theta}(r, 0) = 0$$

where $\Omega(t)$ is an arbitrary function for the rotational velocity of the cylinder.


Both Euler forward and RK4 are implemented.


# Build
Dependencies: opencv and eigen3

``````
cmake .
make
``````

# Run
```
./particle_simulation
```
This example uses: $\Omega(t)=3 \text{sin}(\pi{t})$.

![Alt Text](./data/ex.gif)

