# use-case-ML-force-correction

This repository contains the code for the use case "Machine Learning methods for
force correction" of the project "Hybridlabs", task 2.1.

The physics-based correction method is explained in the paper [Development of a
hardware-in-the-loop wind tunnel setup to study the aerodynamic response of
floating offshore wind turbines](https://doi.org/10.5194/wes-11-839-2026). The
data is described in the paper and is accessible upon request to the authors.
The meta-data is available in the project SharePoint, WP2.

In this use case, we want to investigate the potential of machine learning
methods to correct the aerodynamic forces. Free decay cases in each
degree-of-freedom are performed in no-wind (wind speed is zero) and no-rotation
(rotational speed is zero) conditions. Here, the goal is to develop ML-based
force correction model to estimate the measured rotor forces (Ft) from the
measured acceleration (ACCt, ACCb) and/or position (POS).

The ML methods are regression models and can be developed using
[FOWT-ML](https://github.com/hybridlabs-nl/FOWT-ML) library.

## Acknowledgements

The use case is developed in the
[HybridLabs](https://research-software-directory.org/projects/hybridlabs)
project, a collaboration between TUDelft and Netherlands eScience Center.
