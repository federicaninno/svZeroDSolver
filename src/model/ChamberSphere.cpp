// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#include "ChamberSphere.h"

#include "Model.h"

void ChamberSphere::setup_dofs(DOFHandler& dofhandler) {
  Block::setup_dofs_(dofhandler, 5,
                     {"stress", "tau", "volume"});
}

void ChamberSphere::update_constant(SparseSystem& system,
                                    std::vector<double>& parameters) {
  system.E.coeffRef(global_eqn_ids[2], global_var_ids[5]) = 1;
  system.E.coeffRef(global_eqn_ids[3], global_var_ids[6]) = -1;
  system.F.coeffRef(global_eqn_ids[0], global_var_ids[2]) = -1;
  system.F.coeffRef(global_eqn_ids[0], global_var_ids[4]) = 1;
  system.F.coeffRef(global_eqn_ids[1], global_var_ids[4]) = -1;
  system.F.coeffRef(global_eqn_ids[1], global_var_ids[5]) = 1;
  system.F.coeffRef(global_eqn_ids[3], global_var_ids[1]) = 1;
  system.F.coeffRef(global_eqn_ids[3], global_var_ids[3]) = -1;
  system.F.coeffRef(global_eqn_ids[4], global_var_ids[0]) = 1;
  system.F.coeffRef(global_eqn_ids[4], global_var_ids[2]) = -1;
}

void ChamberSphere::update_time(SparseSystem& system,
                                std::vector<double>& parameters) {
  const double gamma_sigma_max = parameters[global_param_ids[ParamId::gamma_sigma_max]];

  // active stress
  get_elastance_values(parameters);
  system.F.coeffRef(global_eqn_ids[2], global_var_ids[5]) = act;
  system.C.coeffRef(global_eqn_ids[2]) = -act_plus*gamma_sigma_max;
}

void ChamberSphere::update_solution(
    SparseSystem& system, std::vector<double>& parameters,
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& y,
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& dy) {
  const double prestress = parameters[global_param_ids[ParamId::prestress]];
  const double volume0 = parameters[global_param_ids[ParamId::volume0]];
  const double gamma_W1 = parameters[global_param_ids[ParamId::gamma_W1]];
  const double Pout = y[global_var_ids[2]];
  const double stress = y[global_var_ids[4]];
  const double volume = y[global_var_ids[6]];
  system.C.coeffRef(global_eqn_ids[0]) = -Pout*pow((volume + volume0)/volume0, 0.66666666666666663) + Pout + stress*pow((volume + volume0)/volume0, 0.33333333333333331) - stress;
  system.C.coeffRef(global_eqn_ids[1]) = -4*gamma_W1*pow(volume/volume0 + 1, -2.0) + 4*gamma_W1 + prestress;
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[2]) = 1 - pow((volume + volume0)/volume0, 0.66666666666666663);
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[4]) = pow((volume + volume0)/volume0, 0.33333333333333331) - 1;
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[6]) = (-0.66666666666666663*Pout*pow((volume + volume0)/volume0, 0.66666666666666663) + 0.33333333333333331*stress*pow((volume + volume0)/volume0, 0.33333333333333331))/(volume + volume0);
  system.dC_dy.coeffRef(global_eqn_ids[1], global_var_ids[6]) = 8.0*gamma_W1*pow((volume + volume0)/volume0, -3.0)/volume0;
}

void ChamberSphere::get_elastance_values(std::vector<double>& parameters) {
  const double alpha_max = parameters[global_param_ids[ParamId::alpha_max]];
  const double alpha_min = parameters[global_param_ids[ParamId::alpha_min]];
  const double tsys = parameters[global_param_ids[ParamId::tsys]];
  const double tdias = parameters[global_param_ids[ParamId::tdias]];
  const double steepness = parameters[global_param_ids[ParamId::steepness]];

  const double t = model->time;

  const auto T_cardiac = model->cardiac_cycle_period;
  const auto t_in_cycle = fmod(model->time, T_cardiac);

  const double S_plus = 0.5 * (1.0 + tanh((t_in_cycle - tsys) / steepness));
  const double S_minus = 0.5 * (1.0 - tanh((t_in_cycle - tdias) / steepness));

  // indicator function
  const double f = S_plus * S_minus;

  // activation rates
  const double act_t = alpha_max * f + alpha_min * (1 - f);

  act = std::abs(act_t);
  act_plus = std::max(act_t, 0.0);
}