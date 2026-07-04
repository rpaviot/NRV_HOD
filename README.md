# NRVpy


**NRV stands for Nextgen Realisations of Virialized structures.**
This is an HOD code wrapped around an emulator that can performs halo occupation fit 
to projected clustering and galaxy-galaxy lensing 2-point functions.

The code is divided in two parts:

**1) Semi-analytical models of galaxy-halo connection**
This model included modelling of Dsigma, w_theta (wp).
This class is based upon the pyccl library. It includes point mass, and scaling of concentration relation. Lensing magnification contribution will be add shortly.


**2) Numerical models of galaxy-halo connection**
The user can provided two parquet catalogues (sub-halo and particles) in the periodic box to populate halos given a occupation model.
While this technique only provides Dsigma_gm (at the effective redshift of the lens sample), magnification contribution can also
be add to the model analytically. Also, an sub-catalogue such as RockStar provides central halo properties that can allow us to:
- perform assembly bias along any numerical properties such as merger_time, concentration ...
- add local density or anisotropies as additional dependences on halo occupation.
- performs satellites positions assignments following: spherical NFW, elliptical NFW, sub-halo positions.

