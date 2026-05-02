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
  const double n = parameters[global_param_ids[ParamId::n]];
  const double eta = parameters[global_param_ids[ParamId::eta]];
  const double gamma = parameters[global_param_ids[ParamId::gamma]];
  system.E.coeffRef(global_eqn_ids[1], global_var_ids[6]) = 2.0*eta/(n*volume0);
  system.E.coeffRef(global_eqn_ids[2], global_var_ids[5]) = 1;
  system.E.coeffRef(global_eqn_ids[3], global_var_ids[6]) = -1;
  system.F.coeffRef(global_eqn_ids[0], global_var_ids[2]) = -1;
  system.F.coeffRef(global_eqn_ids[0], global_var_ids[4]) = gamma;
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
    
  const double b = parameters[global_param_ids[ParamId::b]];
  const double gamma = parameters[global_param_ids[ParamId::gamma]];
  const double b4s = parameters[global_param_ids[ParamId::b4s]];
  const double a4f = parameters[global_param_ids[ParamId::a4f]];
  const double a4s = parameters[global_param_ids[ParamId::a4s]];
  const double eta = parameters[global_param_ids[ParamId::eta]];
  const double a = parameters[global_param_ids[ParamId::a]];
  const double volume0 = parameters[global_param_ids[ParamId::volume0]];
  const double n = parameters[global_param_ids[ParamId::n]];
  const double b4f = parameters[global_param_ids[ParamId::b4f]];
  const double stress = y[global_var_ids[4]];
  const double volume = y[global_var_ids[6]];
  const double Pout = y[global_var_ids[2]];
  const double dvolume_dt = dy[global_var_ids[6]];
  system.C.coeffRef(global_eqn_ids[0]) = -Pout*pow((volume + volume0)/volume0, (2.0/3.0)/n) + Pout + gamma*stress*pow((volume + volume0)/volume0, (1.0/3.0)/n) - gamma*stress;
  system.C.coeffRef(global_eqn_ids[1]) = (1.0/3.0)*pow((volume + volume0)/volume0, -(17.0/3.0)/n)*(6.0*a*n*volume0*pow((volume + volume0)/volume0, (11.0/3.0)/n)*(pow((volume + volume0)/volume0, 2/n) - 1)*exp(b*pow((volume + volume0)/volume0, -(4.0/3.0)/n)*(pow((volume + volume0)/volume0, (4.0/3.0)/n)*(2.0*pow((volume + volume0)/volume0, (2.0/3.0)/n) - 3.0) + 1.0)) - 6.0*dvolume_dt*eta*pow((volume + volume0)/volume0, (17.0/3.0)/n) + 2*dvolume_dt*eta*pow((volume + volume0)/volume0, (1.0/3.0)*(7 - 3*n)/n)*(1.0*pow((volume + volume0)/volume0, 4/n) + 2.0) + 6.0*n*volume0*pow((volume + volume0)/volume0, (17.0/3.0)/n)*(a4f*exp(b4f*pow(fmax(0.0, pow((volume + volume0)/volume0, (2.0/3.0)/n) - 1.0), 2)) + a4s*exp(b4s*pow(fmax(0.0, pow((volume + volume0)/volume0, (2.0/3.0)/n) - 1.0), 2)))*fmax(0.0, pow((volume + volume0)/volume0, (2.0/3.0)/n) - 1.0))/(n*volume0);
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[2]) = 1 - pow((volume + volume0)/volume0, (2.0/3.0)/n);
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[4]) = gamma*(pow((volume + volume0)/volume0, (1.0/3.0)/n) - 1);
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[6]) = (1.0/3.0)*pow((volume + volume0)/volume0, (1.0/3.0)/n)*(-2*Pout*pow((volume + volume0)/volume0, (1.0/3.0)/n) + gamma*stress)/(n*(volume + volume0));
  system.dC_dy.coeffRef(global_eqn_ids[1], global_var_ids[6]) = pow(volume/volume0 + 1, -(17.0/3.0)/n)*(2.6666666666666665*a*b*n*volume0*pow(volume/volume0 + 1, (19.0/3.0)/n)*exp(b*(2.0*pow(volume/volume0 + 1, (2.0/3.0)/n) - 3.0 + 1.0*pow(volume/volume0 + 1, -(4.0/3.0)/n))) - 5.333333333333333*a*b*n*volume0*pow(volume/volume0 + 1, (13.0/3.0)/n)*exp(b*(2.0*pow(volume/volume0 + 1, (2.0/3.0)/n) - 3.0 + 1.0*pow(volume/volume0 + 1, -(4.0/3.0)/n))) + 2.6666666666666665*a*b*n*volume0*pow(volume/volume0 + 1, (7.0/3.0)/n)*exp(b*(2.0*pow(volume/volume0 + 1, (2.0/3.0)/n) - 3.0 + 1.0*pow(volume/volume0 + 1, -(4.0/3.0)/n))) + 4.0*a*n*volume0*pow(volume/volume0 + 1, (11.0/3.0)/n)*exp(b*(2.0*pow(volume/volume0 + 1, (2.0/3.0)/n) - 3.0 + 1.0*pow(volume/volume0 + 1, -(4.0/3.0)/n))) + 2.6666666666666665*a4f*b4f*n*volume0*pow(volume/volume0 + 1, (19.0/3.0)/n)*exp(b4f*pow(fmax(0.0, pow(volume/volume0 + 1, (2.0/3.0)/n) - 1.0), 2))*(((pow(volume/volume0 + 1, (2.0/3.0)/n) - 1.0 < 0) ? (
   0
)
: ((pow(volume/volume0 + 1, (2.0/3.0)/n) - 1.0 == 0) ? (
   1.0/2.0
)
: (
   1
))))*pow(fmax(0.0, pow(volume/volume0 + 1, (2.0/3.0)/n) - 1.0), 2) + 1.3333333333333333*a4f*n*volume0*pow(volume/volume0 + 1, (19.0/3.0)/n)*exp(b4f*pow(fmax(0.0, pow(volume/volume0 + 1, (2.0/3.0)/n) - 1.0), 2))*(((pow(volume/volume0 + 1, (2.0/3.0)/n) - 1.0 < 0) ? (
   0
)
: ((pow(volume/volume0 + 1, (2.0/3.0)/n) - 1.0 == 0) ? (
   1.0/2.0
)
: (
   1
)))) + 2.6666666666666665*a4s*b4s*n*volume0*pow(volume/volume0 + 1, (19.0/3.0)/n)*exp(b4s*pow(fmax(0.0, pow(volume/volume0 + 1, (2.0/3.0)/n) - 1.0), 2))*(((pow(volume/volume0 + 1, (2.0/3.0)/n) - 1.0 < 0) ? (
   0
)
: ((pow(volume/volume0 + 1, (2.0/3.0)/n) - 1.0 == 0) ? (
   1.0/2.0
)
: (
   1
))))*pow(fmax(0.0, pow(volume/volume0 + 1, (2.0/3.0)/n) - 1.0), 2) + 1.3333333333333333*a4s*n*volume0*pow(volume/volume0 + 1, (19.0/3.0)/n)*exp(b4s*pow(fmax(0.0, pow(volume/volume0 + 1, (2.0/3.0)/n) - 1.0), 2))*(((pow(volume/volume0 + 1, (2.0/3.0)/n) - 1.0 < 0) ? (
   0
)
: ((pow(volume/volume0 + 1, (2.0/3.0)/n) - 1.0 == 0) ? (
   1.0/2.0
)
: (
   1
)))) - 1.3333333333333333*dvolume_dt*eta*n*pow(volume/volume0 + 1, -1 + (7.0/3.0)/n) - 0.66666666666666663*dvolume_dt*eta*n*pow(volume/volume0 + 1, -1 + (19.0/3.0)/n) - 4.4444444444444446*dvolume_dt*eta*pow(volume/volume0 + 1, -1 + (7.0/3.0)/n) + 0.44444444444444442*dvolume_dt*eta*pow(volume/volume0 + 1, -1 + (19.0/3.0)/n))/(pow(n, 2)*volume0*(volume + volume0));
  system.dC_dydot.coeffRef(global_eqn_ids[1], global_var_ids[6]) = (1.0/3.0)*eta*pow((volume + volume0)/volume0, -(17.0/3.0)/n)*(-6.0*pow((volume + volume0)/volume0, (17.0/3.0)/n) + pow((volume + volume0)/volume0, (1.0/3.0)*(7 - 3*n)/n)*(2.0*pow((volume + volume0)/volume0, 4/n) + 4.0))/(n*volume0);
  

  // active stress
  system.C.coeffRef(global_eqn_ids[2]) = -act_plus * sigma_max;
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