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
  const double volume0 = parameters[global_param_ids[ParamId::volume0]];
  const double gamma_eta = parameters[global_param_ids[ParamId::gamma_eta]];
  const double n = parameters[global_param_ids[ParamId::n]];
  system.E.coeffRef(global_eqn_ids[1], global_var_ids[6]) = 2*gamma_eta/(n*volume0);
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
  // active stress
  get_elastance_values(parameters);
  system.F.coeffRef(global_eqn_ids[2], global_var_ids[5]) = act;
}

void ChamberSphere::update_solution(
    SparseSystem& system, std::vector<double>& parameters,
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& y,
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& dy) {
    const double gamma_eta = parameters[global_param_ids[ParamId::gamma_eta]];
  const double gamma_W1 = parameters[global_param_ids[ParamId::gamma_W1]];
  const double volume0 = parameters[global_param_ids[ParamId::volume0]];
  const double gamma_W2 = parameters[global_param_ids[ParamId::gamma_W2]];
  const double n = parameters[global_param_ids[ParamId::n]];
  const double volume = y[global_var_ids[6]];
  const double dvolume_dt = dy[global_var_ids[6]];
  const double stress = y[global_var_ids[4]];
  const double Pout = y[global_var_ids[2]];
  system.C.coeffRef(global_eqn_ids[0]) = -Pout*pow((volume + volume0)/volume0, (2.0/3.0)/n) + Pout + stress*pow((volume + volume0)/volume0, (1.0/3.0)/n) - stress;
  system.C.coeffRef(global_eqn_ids[1]) = (2.0/3.0)*pow((volume + volume0)/volume0, -(17.0/3.0)/n)*(-3*dvolume_dt*gamma_eta*pow((volume + volume0)/volume0, (17.0/3.0)/n) + dvolume_dt*gamma_eta*pow((volume + volume0)/volume0, (1.0/3.0)*(7 - 3*n)/n)*(pow((volume + volume0)/volume0, 4/n) + 2) + 6*n*volume0*pow((volume + volume0)/volume0, (11.0/3.0)/n)*(gamma_W1 + gamma_W2*pow((volume + volume0)/volume0, (2.0/3.0)/n))*(pow((volume + volume0)/volume0, 2/n) - 1))/(n*volume0);
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[2]) = 1 - pow((volume + volume0)/volume0, (2.0/3.0)/n);
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[4]) = pow((volume + volume0)/volume0, (1.0/3.0)/n) - 1;
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[6]) = (1.0/3.0)*pow((volume + volume0)/volume0, (1.0/3.0)/n)*(-2*Pout*pow((volume + volume0)/volume0, (1.0/3.0)/n) + stress)/(n*(volume + volume0));
  system.dC_dy.coeffRef(global_eqn_ids[1], global_var_ids[6]) = (2.0/9.0)*pow((volume + volume0)/volume0, -(17.0/3.0)/n)*(-3*dvolume_dt*gamma_eta*n*pow((volume + volume0)/volume0, (19.0/3.0 - n)/n) - 6*dvolume_dt*gamma_eta*n*pow((volume + volume0)/volume0, (1.0/3.0)*(7 - 3*n)/n) - 10*dvolume_dt*gamma_eta*pow((volume + volume0)/volume0, (19.0/3.0 - n)/n) - 20*dvolume_dt*gamma_eta*pow((volume + volume0)/volume0, (1.0/3.0)*(7 - 3*n)/n) + 12*dvolume_dt*gamma_eta*pow((volume + volume0)/volume0, (1.0/3.0)*(19 - 3*n)/n) + 36*gamma_W1*n*volume0*pow((volume + volume0)/volume0, (11.0/3.0)/n) + 12*gamma_W2*n*volume0*pow((volume + volume0)/volume0, (19.0/3.0)/n) + 24*gamma_W2*n*volume0*pow((volume + volume0)/volume0, (13.0/3.0)/n))/(pow(n, 2)*volume0*(volume + volume0));
  system.dC_dydot.coeffRef(global_eqn_ids[1], global_var_ids[6]) = (2.0/3.0)*gamma_eta*pow((volume + volume0)/volume0, -(17.0/3.0)/n)*(-3*pow((volume + volume0)/volume0, (17.0/3.0)/n) + pow((volume + volume0)/volume0, (1.0/3.0)*(7 - 3*n)/n)*(pow((volume + volume0)/volume0, 4/n) + 2))/(n*volume0);

  // active stress
  system.C.coeffRef(global_eqn_ids[2]) = -act_plus*gamma_sigma_max;
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