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
  const double b_t = parameters[global_param_ids[ParamId::b_t]];
  const double b_f = parameters[global_param_ids[ParamId::b_f]];
  const double volume0 = parameters[global_param_ids[ParamId::volume0]];
  const double guccione_C = parameters[global_param_ids[ParamId::guccione_C]];
  const double prestress = parameters[global_param_ids[ParamId::prestress]];
  const double Pout = y[global_var_ids[2]];
  const double volume = y[global_var_ids[6]];
  const double stress = y[global_var_ids[4]];
  system.C.coeffRef(global_eqn_ids[0]) = -Pout*pow((volume + volume0)/volume0, 0.66666666666666663) + Pout + stress*pow((volume + volume0)/volume0, 0.33333333333333331) - stress;
  system.C.coeffRef(global_eqn_ids[1]) = pow((volume + volume0)/volume0, -2.6666666666666665)*(-guccione_C*(0.5*b_t*(1 - pow((volume + volume0)/volume0, 1.3333333333333333)) - 0.25*pow((volume + volume0)/volume0, 3.333333333333333)*(b_f + b_t)*(pow((volume + volume0)/volume0, 0.66666666666666663) - 1))*exp(0.25*pow((volume + volume0)/volume0, -2.6666666666666665)*(b_t*pow(1 - pow((volume + volume0)/volume0, 1.3333333333333333), 2) + pow((volume + volume0)/volume0, 2.6666666666666665)*(b_f + b_t)*pow(pow((volume + volume0)/volume0, 0.66666666666666663) - 1, 2))) + prestress*pow((volume + volume0)/volume0, 2.6666666666666665));
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[2]) = 1 - pow((volume + volume0)/volume0, 0.66666666666666663);
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[4]) = pow((volume + volume0)/volume0, 0.33333333333333331) - 1;
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[6]) = (-0.66666666666666663*Pout*pow((volume + volume0)/volume0, 0.66666666666666663) + 0.33333333333333331*stress*pow((volume + volume0)/volume0, 0.33333333333333331))/(volume + volume0);
  system.dC_dy.coeffRef(global_eqn_ids[1], global_var_ids[6]) = pow((volume + volume0)/volume0, -8.0)*(pow((volume + volume0)/volume0, 2.6666666666666665)*(guccione_C*(0.5*b_t*(pow((volume + volume0)/volume0, 1.3333333333333333) - 1) + 0.25*pow((volume + volume0)/volume0, 3.333333333333333)*(b_f + b_t)*(pow((volume + volume0)/volume0, 0.66666666666666663) - 1))*(0.66666666666666663*b_t*pow((volume + volume0)/volume0, 1.3333333333333333)*(pow((volume + volume0)/volume0, 1.3333333333333333) - 1) - 0.66666666666666663*b_t*pow(pow((volume + volume0)/volume0, 1.3333333333333333) - 1, 2) + 0.33333333333333331*pow((volume + volume0)/volume0, 3.333333333333333)*(b_f + b_t)*(pow((volume + volume0)/volume0, 0.66666666666666663) - 1))*exp(0.25*pow((volume + volume0)/volume0, -2.6666666666666665)*(b_t*pow(pow((volume + volume0)/volume0, 1.3333333333333333) - 1, 2) + pow((volume + volume0)/volume0, 2.6666666666666665)*(b_f + b_t)*pow(pow((volume + volume0)/volume0, 0.66666666666666663) - 1, 2))) + pow((volume + volume0)/volume0, 2.6666666666666665)*(guccione_C*(0.66666666666666663*b_t*pow((volume + volume0)/volume0, 1.3333333333333333) + 0.83333333333333326*pow((volume + volume0)/volume0, 3.333333333333333)*(b_f + b_t)*(pow((volume + volume0)/volume0, 0.66666666666666663) - 1) + 0.16666666666666666*pow((volume + volume0)/volume0, 3.9999999999999996)*(b_f + b_t))*exp(0.25*pow((volume + volume0)/volume0, -2.6666666666666665)*(b_t*pow(pow((volume + volume0)/volume0, 1.3333333333333333) - 1, 2) + pow((volume + volume0)/volume0, 2.6666666666666665)*(b_f + b_t)*pow(pow((volume + volume0)/volume0, 0.66666666666666663) - 1, 2))) + 2.6666666666666665*prestress*pow((volume + volume0)/volume0, 2.6666666666666665))) - 2.6666666666666665*pow((volume + volume0)/volume0, 5.333333333333333)*(guccione_C*(0.5*b_t*(pow((volume + volume0)/volume0, 1.3333333333333333) - 1) + 0.25*pow((volume + volume0)/volume0, 3.333333333333333)*(b_f + b_t)*(pow((volume + volume0)/volume0, 0.66666666666666663) - 1))*exp(0.25*pow((volume + volume0)/volume0, -2.6666666666666665)*(b_t*pow(pow((volume + volume0)/volume0, 1.3333333333333333) - 1, 2) + pow((volume + volume0)/volume0, 2.6666666666666665)*(b_f + b_t)*pow(pow((volume + volume0)/volume0, 0.66666666666666663) - 1, 2))) + prestress*pow((volume + volume0)/volume0, 2.6666666666666665)))/(volume + volume0);
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

void ChamberSphere::update_gradient(
    Eigen::SparseMatrix<double>& jacobian,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& residual,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& alpha, std::vector<double>& y,
    std::vector<double>& dy) {
  // Guccione passive law; generated by scripts/jacobian.py from scripts/ChamberSphere.yaml.
  const double volume0 = alpha[global_param_ids[ParamId::volume0]];
  const double guccione_C = alpha[global_param_ids[ParamId::guccione_C]];
  const double gamma_sigma_max = alpha[global_param_ids[ParamId::gamma_sigma_max]];
  const double prestress = alpha[global_param_ids[ParamId::prestress]];
  const double alpha_max = alpha[global_param_ids[ParamId::alpha_max]];
  const double alpha_min = alpha[global_param_ids[ParamId::alpha_min]];
  const double tsys = alpha[global_param_ids[ParamId::tsys]];
  const double tdias = alpha[global_param_ids[ParamId::tdias]];
  const double steepness = alpha[global_param_ids[ParamId::steepness]];
  const double b_f = alpha[global_param_ids[ParamId::b_f]];
  const double b_t = alpha[global_param_ids[ParamId::b_t]];
  const double Pin = y[global_var_ids[0]];
  const double Qin = y[global_var_ids[1]];
  const double Pout = y[global_var_ids[2]];
  const double Qout = y[global_var_ids[3]];
  const double stress = y[global_var_ids[4]];
  const double tau = y[global_var_ids[5]];
  const double volume = y[global_var_ids[6]];
  const double dtau_dt = dy[global_var_ids[5]];
  const double dvolume_dt = dy[global_var_ids[6]];
  const double t = model->cardiac_cycle_period > 0.0 ? fmod(model->time, model->cardiac_cycle_period) : model->time;

  const double x0 = 1.0/volume0;
  const double x1 = volume + volume0;
  const double x2 = x0*x1;
  const double x3 = pow(x2, 0.66666666666666663);
  const double x4 = Pout*x3;
  const double x5 = pow(x2, 0.33333333333333331);
  const double x6 = pow(x2, 1.3333333333333333);
  const double x7 = 1.0/x6;
  const double x8 = 0.5*x7 - 0.5;
  const double x9 = b_f + b_t;
  const double x10 = 0.5*x3 - 0.5;
  const double x11 = 0.5*b_t;
  const double x12 = 1.0/steepness;
  const double x13 = t - tdias;
  const double x14 = tanh(x12*x13);
  const double x15 = 0.5*x14 - 0.5;
  const double x16 = t - tsys;
  const double x17 = tanh(x12*x16);
  const double x18 = 0.5*x17 + 0.5;
  const double x19 = x15*x18;
  const double x20 = x19 + 1;
  const double x21 = alpha_max*x19 - alpha_min*x20;
  const double x22 = -x15;
  const double x23 = x18*x22;
  const double x24 = 1 - x23;
  const double x25 = alpha_max*x23 + alpha_min*x24;
  const double x26 = fmax(0, x25);
  const double x27 = volume*x0/x1;
  const double x28 = pow(x2, 5.333333333333333);
  const double x29 = 0.66666666666666663*b_t;
  const double x30 = pow(x2, 2.6666666666666665);
  const double x31 = x3 - 1;
  const double x32 = 1 - x6;
  const double x33 = pow(x2, 3.333333333333333);
  const double x34 = x11*x32 - 0.25*x31*x33*x9;
  const double x35 = x30*x34;
  const double x36 = pow(x32, 2);
  const double x37 = x30*pow(x31, 2);
  const double x38 = 1.0/x30;
  const double x39 = 0.25*x38;
  const double x40 = exp(x39*(b_t*x36 + x37*x9));
  const double x41 = guccione_C*x40;
  const double x42 = pow(x2, -8.0)*x41;
  const double x43 = gamma_sigma_max*(x25 > 0 ? 1.0 : 0.0);
  const double x44 = (((x21) > 0) - ((x21) < 0));
  const double x45 = 1 - pow(x17, 2);
  const double x46 = 0.5*x12;
  const double x47 = x45*x46;
  const double x48 = 1 - pow(x14, 2);
  const double x49 = x18*x48;
  const double x50 = alpha_max*x46*x49 - 0.5*alpha_min*x12*x18*x48;
  const double x51 = 0.5/pow(steepness, 2);
  const double x52 = x16*x45*x51;
  const double x53 = x22*x52;
  const double x54 = x13*x49*x51;
  const double x55 = alpha_max*x54;
  const double x56 = x15*x52;

  residual(global_eqn_ids[0]) = stress*x5 - x4;
  residual(global_eqn_ids[1]) = guccione_C*(-b_t*x7*x8 + x10*x3*(0.5*b_f + x11))*exp(b_t*pow(x8, 2) + pow(x10, 2)*x9) + prestress - stress + tau;
  residual(global_eqn_ids[2]) = dtau_dt - gamma_sigma_max*x26 + tau*fabs(x21);
  residual(global_eqn_ids[3]) = Qin - Qout - dvolume_dt;
  residual(global_eqn_ids[4]) = Pin - Pout;
  jacobian.coeffRef(global_eqn_ids[0], global_param_ids[ParamId::volume0]) = x27*(-0.33333333333333331*stress*x5 + 0.66666666666666663*x4);
  jacobian.coeffRef(global_eqn_ids[1], global_param_ids[ParamId::volume0]) = x27*x42*(x28*(x29*(x6 - 2) + 0.16666666666666666*x30*x9*(-x3*x31 - x6)) + x35*(-x29*x32 + 0.33333333333333331*x31*x33*x9));
  jacobian.coeffRef(global_eqn_ids[1], global_param_ids[ParamId::guccione_C]) = -x34*x38*x40;
  jacobian.coeffRef(global_eqn_ids[2], global_param_ids[ParamId::gamma_sigma_max]) = -x26;
  jacobian.coeffRef(global_eqn_ids[1], global_param_ids[ParamId::prestress]) = 1;
  jacobian.coeffRef(global_eqn_ids[2], global_param_ids[ParamId::alpha_max]) = tau*x15*x18*x44 - x23*x43;
  jacobian.coeffRef(global_eqn_ids[2], global_param_ids[ParamId::alpha_min]) = -tau*x20*x44 - x24*x43;
  jacobian.coeffRef(global_eqn_ids[2], global_param_ids[ParamId::tsys]) = tau*x44*(-alpha_max*x15*x47 + 0.5*alpha_min*x12*x15*x45) - x43*(-alpha_max*x22*x47 + 0.5*alpha_min*x12*x22*x45);
  jacobian.coeffRef(global_eqn_ids[2], global_param_ids[ParamId::tdias]) = -tau*x44*x50 - x43*x50;
  jacobian.coeffRef(global_eqn_ids[2], global_param_ids[ParamId::steepness]) = tau*x44*(-alpha_max*x56 - alpha_min*(-x54 - x56) - x55) - x43*(-alpha_max*x53 + alpha_min*(x53 - x54) + x55);
  jacobian.coeffRef(global_eqn_ids[1], global_param_ids[ParamId::b_f]) = x31*x39*x41*(-x31*x34 + x33);
  jacobian.coeffRef(global_eqn_ids[1], global_param_ids[ParamId::b_t]) = x42*(x28*(x33*(0.25*x3 - 0.25) + 0.5*x6 - 0.5) - 0.25*x35*(x36 + x37));
}