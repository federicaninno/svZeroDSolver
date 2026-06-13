// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause
/**
 * @file ChamberSphere.h
 * @brief model::ChamberSphere source file
 */
#ifndef SVZERODSOLVER_MODEL_ChamberSphere_HPP_
#define SVZERODSOLVER_MODEL_ChamberSphere_HPP_

#include <math.h>

#include "Block.h"
#include "SparseSystem.h"

/**
 * @brief Spherical heart chamber model
 *
 * Models the mechanical behavior of a spherical heart chamber with active
 * contraction. For reference, see \cite caruel13 Equations (13a-b) for
 * continuum mechanics (without length-dependent contraction valves, vessels)
 * and \cite pfaller2019importance Equations (12-16) for the simplified active
 * contraction model.
 *
 * ### Helper Functions
 *
 * Cauchy-Green deformation tensor and time derivative:
 * \f[
 * C = \left(1 + \frac{r}{r_0} \right)^2
 * \f]
 * \f[
 * \dot{C} = 2 \left(1 + \frac{r}{r_0} \right) \frac{\dot{r}}{r_0}
 * \f]
 *
 * ### Governing equations
 *
 * 1. Balance of linear momentum:
 * \f[
 * \rho d_0 \dot{v} + \frac{d_0}{r_0} \left(1 + \frac{r}{r_0} \right) S -
 P_\text{out} C = 0
 * \f]
 *
 * 2. Spherical stress (shape correction factor \f$n = 1\f$, no viscosity), with
 * a Guccione passive wall stress \f$S_\text{pas}\f$:
 * \f[
 * -S + \tau + S_\text{pas} = 0, \quad
 * S_\text{pas} = C e^{Q} \left[ \frac{b_f + b_t}{2} \lambda^2 E_p
 * - b_t \lambda^{-4} E_r \right] + \text{prestress}
 * \f]
 * with \f$\lambda^2 = C_G\f$, in-plane/radial Green-Lagrange strains
 * \f$E_p = (\lambda^2 - 1)/2\f$, \f$E_r = (\lambda^{-4} - 1)/2\f$ and
 * \f$Q = (b_f + b_t) E_p^2 + b_t E_r^2\f$. The shear exponent \f$b_{fs}\f$ drops
 * out because the equibiaxial spherical deformation has no shear.
 *
 * 3. Volume change:
 * \f[
 * 4 \pi r_0^2 Cv - \dot{V} = 0
 * \f]
 *
 * 4. Active stress:
 * \f[
 * \dot{\tau} + a \tau - \sigma_\text{max} a_+ = 0, \quad a_+ = \max(a, 0),
 \quad a = f\alpha_\text{max} + (1 - f)\alpha_\text{min}
 * \f]
 * with indicator function
 * \f[
 * f = S_+ \cdot S_-, \quad S_\pm = \frac{1}{2} \left(1.0 \pm \text{tanh}\left(
 \frac{t - t_\text{sys/dias}} {\gamma} \right) \right)
 * \f]
 *
 * 5. Acceleration:
 * \f[
 * \dot{r} - v = 0
 * \f]
 *
 * 6. Conservation of mass:
 * \f[
 * Q_\text{in} - Q_\text{out} - \dot{V} = 0
 * \f]
 *
 * 7. Pressure equality:
 * \f[
 * P_\text{in} - P_\text{out} = 0
 * \f]
 *
 * ### Parameters
 *
 * Parameter sequence for constructing this block:
 *
 * * `volume0` - Reference (unloaded) chamber volume \f$V_0\f$
 * * `guccione_C` - Scaled Guccione passive scaling \f$\gamma C\f$
 * * `gamma_sigma_max` - Scaled maximum active stress \f$\gamma \sigma_\text{max}\f$
 * * `prestress` - Prestress
 * * `alpha_max` - Maximum activation parameter \f$\alpha_\text{max}\f$
 * * `alpha_min` - Minimum activation parameter \f$\alpha_\text{min}\f$
 * * `tsys` - Systole timing parameter \f$t_\text{sys}\f$
 * * `tdias` - Diastole timing parameter \f$t_\text{dias}\f$
 * * `steepness` - Activation steepness parameter
 * * `b_f` - Guccione fiber exponent (dimensionless)
 * * `b_t` - Guccione transverse exponent (dimensionless)
 *
 * ### Usage in json configuration file
 *
 *     "vessels": [
 *        {
 *            "boundary_conditions": {},
 *            "vessel_id": 1,
 *            "vessel_length": 1.0,
 *            "vessel_name": "ventricle",
 *            "zero_d_element_type": "ChamberSphere",
 *            "zero_d_element_values": {
 *                "volume0" : 1e-4,
 *                "guccione_C" : 1e3,
 *                "gamma_sigma_max" : 185e3,
 *                "prestress" : 0.0,
 *                "alpha_max": 30.0,
 *                "alpha_min": -30.0,
 *                "tsys": 0.170,
 *                "tdias": 0.484,
 *                "steepness": 0.005,
 *                "b_f": 8.0,
 *                "b_t": 3.0
 *            }
 *        }
 *     ]
 *
 * ### Internal variables
 *
 * Names of internal variables in this block's output:
 *
 * * `radius` - Chamber radius \f$r\f$
 * * `velo` - Chamber velocity \f$\dot{r}\f$
 * * `stress` - Spherical stress \f$S\f$
 * * `tau` - Active stress \f$\tau\f$
 * * `volume` - Chamber volume \f$V\f$
 *
 */
class ChamberSphere : public Block {
 public:
  /**
   * @brief Local IDs of the parameters
   *
   */
  enum ParamId {
    volume0 = 0,
    guccione_C = 1,
    gamma_sigma_max = 2,
    prestress = 3,
    alpha_max = 4,
    alpha_min = 5,
    tsys = 6,
    tdias = 7,
    steepness = 8,
    b_f = 9,
    b_t = 10
  };

  /**
   * @brief Construct a new ChamberSphere object
   *
   * @param id Global ID of the block
   * @param model The model to which the block belongs
   */
  ChamberSphere(int id, Model* model)
      : Block(id, model, BlockType::chamber_sphere, BlockClass::vessel,
              {{"volume0", InputParameter()},
               {"guccione_C", InputParameter()},
               {"gamma_sigma_max", InputParameter()},
               {"prestress", InputParameter()},
               {"alpha_max", InputParameter()},
               {"alpha_min", InputParameter()},
               {"tsys", InputParameter()},
               {"tdias", InputParameter()},
               {"steepness", InputParameter()},
               {"b_f", InputParameter()},
               {"b_t", InputParameter()}}) {}

  /**
   * @brief Set up the degrees of freedom (DOF) of the block
   *
   * Set \ref global_var_ids and \ref global_eqn_ids of the element based on the
   * number of equations and the number of internal variables of the
   * element.
   *
   * @param dofhandler Degree-of-freedom handler to register variables and
   * equations at
   */
  void setup_dofs(DOFHandler& dofhandler);

  /**
   * @brief Update the constant contributions of the element in a sparse
   system
   *
   * @param system System to update contributions at
   * @param parameters Parameters of the model
   */
  void update_constant(SparseSystem& system, std::vector<double>& parameters);

  /**
   * @brief Update the time-dependent contributions of the element in a sparse
   * system
   *
   * @param system System to update contributions at
   * @param parameters Parameters of the model
   */
  void update_time(SparseSystem& system, std::vector<double>& parameters);

  /**
   * @brief Update the solution-dependent contributions of the element in a
   * sparse system
   *
   * @param system System to update contributions at
   * @param parameters Parameters of the model
   * @param y Current solution
   * @param dy Current derivate of the solution
   */
  void update_solution(SparseSystem& system, std::vector<double>& parameters,
                       const Eigen::Matrix<double, Eigen::Dynamic, 1>& y,
                       const Eigen::Matrix<double, Eigen::Dynamic, 1>& dy);

  /**
   * @brief Update the elastance functions which depend on time
   *
   * @param parameters Parameters of the model
   */
  void get_elastance_values(std::vector<double>& parameters);

  /**
   * @brief Set the gradient of the block contributions with respect to the
   * parameters
   *
   * Calibrates the chamber from a full-state observation set (the data carry the
   * internal variables stress, tau, volume and their derivatives). The
   * time-independent parameters (volume0, gamma_W1, prestress) appear in the
   * momentum and spherical-stress equations, which are pure functions of the
   * state. The active-stress equation (and its parameters gamma_sigma_max,
   * alpha_max, alpha_min, tsys, tdias, steepness) depends on the observation
   * time, supplied by the optimizer via ``model->time`` when a time vector is
   * given; without it, only the time-independent parameters are identifiable.
   * Residual/Jacobian expressions are derived symbolically (cf.
   * scripts/jacobian.py from scripts/ChamberSphere.yaml).
   *
   * @param jacobian Jacobian with respect to the parameters
   * @param residual Residual with respect to the parameters
   * @param alpha Current parameter vector
   * @param y Current solution
   * @param dy Time-derivative of the current solution
   */
  void update_gradient(
      Eigen::SparseMatrix<double>& jacobian,
      Eigen::Matrix<double, Eigen::Dynamic, 1>& residual,
      Eigen::Matrix<double, Eigen::Dynamic, 1>& alpha, std::vector<double>& y,
      std::vector<double>& dy) override;

 private:
  double act = 0.0;       // activation function
  double act_plus = 0.0;  // act_plus = max(act, 0)

  /**
   * @brief Number of triplets of element
   *
   * Number of triplets that the element contributes to the global system
   * (relevant for sparse memory reservation)
   */
  TripletsContributions num_triplets{9, 2, 4};
};

#endif  // SVZERODSOLVER_MODEL_ChamberSphere_HPP_
