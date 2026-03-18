// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#include "ChamberSphere.h"

#include "Model.h"

void ChamberSphere::setup_dofs(DOFHandler& dofhandler) {
  Block::setup_dofs_(dofhandler, 7,
                     {"radius", "velo", "stress", "tau", "volume"});
}

void ChamberSphere::update_constant(SparseSystem& system,
                                    std::vector<double>& parameters) {
    const double eta = parameters[global_param_ids[ParamId::eta]];
  const double thick0 = parameters[global_param_ids[ParamId::thick0]];
  const double rho = parameters[global_param_ids[ParamId::rho]];
  const double radius0 = parameters[global_param_ids[ParamId::radius0]];
  system.E.coeffRef(global_eqn_ids[0], global_var_ids[5]) = rho*thick0;
  system.E.coeffRef(global_eqn_ids[1], global_var_ids[4]) = 6*eta/radius0;
  system.E.coeffRef(global_eqn_ids[2], global_var_ids[8]) = -1;
  system.E.coeffRef(global_eqn_ids[3], global_var_ids[7]) = 1;
  system.E.coeffRef(global_eqn_ids[4], global_var_ids[4]) = 1;
  system.E.coeffRef(global_eqn_ids[5], global_var_ids[8]) = -1;
  system.F.coeffRef(global_eqn_ids[0], global_var_ids[2]) = -1;
  system.F.coeffRef(global_eqn_ids[0], global_var_ids[6]) = thick0/radius0;
  system.F.coeffRef(global_eqn_ids[1], global_var_ids[6]) = -1;
  system.F.coeffRef(global_eqn_ids[1], global_var_ids[7]) = 1;
  system.F.coeffRef(global_eqn_ids[2], global_var_ids[5]) = 4*M_PI*pow(radius0, 2);
  system.F.coeffRef(global_eqn_ids[4], global_var_ids[5]) = -1;
  system.F.coeffRef(global_eqn_ids[5], global_var_ids[1]) = 1;
  system.F.coeffRef(global_eqn_ids[5], global_var_ids[3]) = -1;
  system.F.coeffRef(global_eqn_ids[6], global_var_ids[0]) = 1;
  system.F.coeffRef(global_eqn_ids[6], global_var_ids[2]) = -1;
}

void ChamberSphere::update_time(SparseSystem& system,
                                std::vector<double>& parameters) {
  // active stress
  get_elastance_values(parameters);
  system.F.coeffRef(global_eqn_ids[3], global_var_ids[7]) = act;
}

void ChamberSphere::update_solution(
    SparseSystem& system, std::vector<double>& parameters,
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& y,
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& dy) {
    
  const double eta = parameters[global_param_ids[ParamId::eta]];
  const double W2 = parameters[global_param_ids[ParamId::W2]];
  const double radius0 = parameters[global_param_ids[ParamId::radius0]];
  const double W1 = parameters[global_param_ids[ParamId::W1]];
  const double thick0 = parameters[global_param_ids[ParamId::thick0]];
  const double dradius_dt = dy[global_var_ids[4]];
  const double Pout = y[global_var_ids[2]];
  const double stress = y[global_var_ids[6]];
  const double velo = y[global_var_ids[5]];
  const double radius = y[global_var_ids[4]];
  const double sigma_max = parameters[global_param_ids[ParamId::sigma_max]];
  system.C.coeffRef(global_eqn_ids[0]) = radius*(-Pout*radius - 2*Pout*radius0 + stress*thick0)/pow(radius0, 2);
  system.C.coeffRef(global_eqn_ids[1]) = (2.0*W1*pow(radius0, 2)*pow(radius + radius0, 5)*(-pow(radius0, 6) + pow(radius + radius0, 6))*exp(W2*(1.0*pow(radius0, 6) - 3.0*pow(radius0, 2)*pow(radius + radius0, 4) + 2.0*pow(radius + radius0, 6))/(pow(radius0, 2)*pow(radius + radius0, 4))) - 6*dradius_dt*eta*radius0*pow(radius + radius0, 11) + 2*dradius_dt*eta*(2*pow(radius0, 12) + pow(radius + radius0, 12)))/(pow(radius0, 2)*pow(radius + radius0, 11));
  system.C.coeffRef(global_eqn_ids[2]) = 4*M_PI*radius*velo*(radius + 2*radius0);
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[2]) = -radius*(radius + 2*radius0)/pow(radius0, 2);
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[4]) = (-2*Pout*radius - 2*Pout*radius0 + stress*thick0)/pow(radius0, 2);
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[6]) = radius*thick0/pow(radius0, 2);
  system.dC_dy.coeffRef(global_eqn_ids[1], global_var_ids[4]) = (22.0*W1*pow(radius0, 2)*pow(radius + radius0, 5)*(pow(radius0, 6) - pow(radius + radius0, 6))*exp(W2*(1.0*pow(radius0, 6) - 3.0*pow(radius0, 2)*pow(radius + radius0, 4) + 2.0*pow(radius + radius0, 6))/(pow(radius0, 2)*pow(radius + radius0, 4))) + 66*dradius_dt*eta*radius0*pow(radius + radius0, 11) - 22*dradius_dt*eta*(2*pow(radius0, 12) + pow(radius + radius0, 12)) + (radius + radius0)*(2.0*W1*W2*(pow(radius0, 6) - pow(radius + radius0, 6))*(4.0*pow(radius0, 6) - 12.0*pow(radius0, 2)*pow(radius + radius0, 4) + 8.0*pow(radius + radius0, 6) + 12.0*pow(radius + radius0, 4)*(pow(radius0, 2) - pow(radius + radius0, 2)))*exp(W2*(1.0*pow(radius0, 6) - 3.0*pow(radius0, 2)*pow(radius + radius0, 4) + 2.0*pow(radius + radius0, 6))/(pow(radius0, 2)*pow(radius + radius0, 4))) + pow(radius + radius0, 4)*(12.0*W1*pow(radius0, 2)*pow(radius + radius0, 6)*exp(W2*(1.0*pow(radius0, 6) - 3.0*pow(radius0, 2)*pow(radius + radius0, 4) + 2.0*pow(radius + radius0, 6))/(pow(radius0, 2)*pow(radius + radius0, 4))) - 10.0*W1*pow(radius0, 2)*(pow(radius0, 6) - pow(radius + radius0, 6))*exp(W2*(1.0*pow(radius0, 6) - 3.0*pow(radius0, 2)*pow(radius + radius0, 4) + 2.0*pow(radius + radius0, 6))/(pow(radius0, 2)*pow(radius + radius0, 4))) - 66*dradius_dt*eta*radius0*pow(radius + radius0, 6) + 24*dradius_dt*eta*pow(radius + radius0, 7))))/(pow(radius0, 2)*pow(radius + radius0, 12));
  system.dC_dy.coeffRef(global_eqn_ids[2], global_var_ids[4]) = 8*M_PI*velo*(radius + radius0);
  system.dC_dy.coeffRef(global_eqn_ids[2], global_var_ids[5]) = 4*M_PI*radius*(radius + 2*radius0);
  system.dC_dydot.coeffRef(global_eqn_ids[1], global_var_ids[4]) = 2*eta*radius/pow(radius0, 2) + 4*eta*pow(radius0, 10)/pow(radius + radius0, 11) - 4*eta/radius0;

  // active stress
  system.C.coeffRef(global_eqn_ids[3]) = -act_plus * sigma_max;
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