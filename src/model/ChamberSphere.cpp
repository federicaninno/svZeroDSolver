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
  system.E.coeffRef(global_eqn_ids[3], global_var_ids[6]) = -1;
  system.F.coeffRef(global_eqn_ids[0], global_var_ids[2]) = -1;
  system.F.coeffRef(global_eqn_ids[0], global_var_ids[4]) = 1;
  system.F.coeffRef(global_eqn_ids[1], global_var_ids[4]) = -1;
  system.F.coeffRef(global_eqn_ids[1], global_var_ids[5]) = 1;
  system.F.coeffRef(global_eqn_ids[2], global_var_ids[5]) = 1;
  system.F.coeffRef(global_eqn_ids[3], global_var_ids[1]) = 1;
  system.F.coeffRef(global_eqn_ids[3], global_var_ids[3]) = -1;
  system.F.coeffRef(global_eqn_ids[4], global_var_ids[0]) = 1;
  system.F.coeffRef(global_eqn_ids[4], global_var_ids[2]) = -1;
}

void ChamberSphere::update_time(SparseSystem& system,
                                std::vector<double>& parameters) {
  // active stress is the algebraic two-hill twitch tau = gamma_sigma_max A(t)
  const double t_shift = parameters[global_param_ids[ParamId::t_shift]];
  const double m2 = parameters[global_param_ids[ParamId::m2]];
  const double m1 = parameters[global_param_ids[ParamId::m1]];
  const double tau_1 = parameters[global_param_ids[ParamId::tau_1]];
  const double gamma_sigma_max = parameters[global_param_ids[ParamId::gamma_sigma_max]];
  const double tau_2 = parameters[global_param_ids[ParamId::tau_2]];
  const double t = model->cardiac_cycle_period > 0.0 ? fmod(model->time, model->cardiac_cycle_period) : model->time;
  system.C.coeffRef(global_eqn_ids[2]) = -gamma_sigma_max*pow((t - t_shift)/tau_1, m1)/((pow((t - t_shift)/tau_1, m1) + 1)*(pow((t - t_shift)/tau_2, m2) + 1));
}

void ChamberSphere::update_solution(
    SparseSystem& system, std::vector<double>& parameters,
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& y,
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& dy) {
  const double b_f = parameters[global_param_ids[ParamId::b_f]];
  const double b_t = parameters[global_param_ids[ParamId::b_t]];
  const double volume0 = parameters[global_param_ids[ParamId::volume0]];
  const double guccione_C = parameters[global_param_ids[ParamId::guccione_C]];
  const double volume = y[global_var_ids[6]];
  const double stress = y[global_var_ids[4]];
  const double Pout = y[global_var_ids[2]];
  system.C.coeffRef(global_eqn_ids[0]) = -Pout*pow((volume + volume0)/volume0, 0.66666666666666663) + Pout + stress*pow((volume + volume0)/volume0, 0.33333333333333331) - stress;
  system.C.coeffRef(global_eqn_ids[1]) = guccione_C*pow((volume + volume0)/volume0, -2.6666666666666665)*(0.5*b_t*(pow((volume + volume0)/volume0, 1.3333333333333333) - 1) + 0.25*pow((volume + volume0)/volume0, 3.333333333333333)*(b_f + b_t)*(pow((volume + volume0)/volume0, 0.66666666666666663) - 1))*exp(0.25*pow((volume + volume0)/volume0, -2.6666666666666665)*(b_t*pow(pow((volume + volume0)/volume0, 1.3333333333333333) - 1, 2) + pow((volume + volume0)/volume0, 2.6666666666666665)*(b_f + b_t)*pow(pow((volume + volume0)/volume0, 0.66666666666666663) - 1, 2)));
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[2]) = 1 - pow((volume + volume0)/volume0, 0.66666666666666663);
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[4]) = pow((volume + volume0)/volume0, 0.33333333333333331) - 1;
  system.dC_dy.coeffRef(global_eqn_ids[0], global_var_ids[6]) = (-0.66666666666666663*Pout*pow((volume + volume0)/volume0, 0.66666666666666663) + 0.33333333333333331*stress*pow((volume + volume0)/volume0, 0.33333333333333331))/(volume + volume0);
  system.dC_dy.coeffRef(global_eqn_ids[1], global_var_ids[6]) = guccione_C*pow((volume + volume0)/volume0, -8.0)*(pow((volume + volume0)/volume0, 2.6666666666666665)*(0.5*b_t*(pow((volume + volume0)/volume0, 1.3333333333333333) - 1) + 0.25*pow((volume + volume0)/volume0, 3.333333333333333)*(b_f + b_t)*(pow((volume + volume0)/volume0, 0.66666666666666663) - 1))*(0.66666666666666663*b_t*pow((volume + volume0)/volume0, 1.3333333333333333)*(pow((volume + volume0)/volume0, 1.3333333333333333) - 1) - 0.66666666666666663*b_t*pow(pow((volume + volume0)/volume0, 1.3333333333333333) - 1, 2) + 0.33333333333333331*pow((volume + volume0)/volume0, 3.333333333333333)*(b_f + b_t)*(pow((volume + volume0)/volume0, 0.66666666666666663) - 1)) + pow((volume + volume0)/volume0, 5.333333333333333)*(0.66666666666666663*b_t*pow((volume + volume0)/volume0, 1.3333333333333333) - 1.3333333333333333*b_t*(pow((volume + volume0)/volume0, 1.3333333333333333) - 1) + 0.16666666666666663*pow((volume + volume0)/volume0, 3.333333333333333)*(b_f + b_t)*(pow((volume + volume0)/volume0, 0.66666666666666663) - 1) + 0.16666666666666666*pow((volume + volume0)/volume0, 3.9999999999999996)*(b_f + b_t)))*exp(0.25*pow((volume + volume0)/volume0, -2.6666666666666665)*(b_t*pow(pow((volume + volume0)/volume0, 1.3333333333333333) - 1, 2) + pow((volume + volume0)/volume0, 2.6666666666666665)*(b_f + b_t)*pow(pow((volume + volume0)/volume0, 0.66666666666666663) - 1, 2)))/(volume + volume0);
}


void ChamberSphere::update_gradient(
    Eigen::SparseMatrix<double>& jacobian,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& residual,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& alpha, std::vector<double>& y,
    std::vector<double>& dy) {
  // Guccione passive + two-hill active twitch; generated by scripts/jacobian.py.
  const double volume0 = alpha[global_param_ids[ParamId::volume0]];
  const double guccione_C = alpha[global_param_ids[ParamId::guccione_C]];
  const double gamma_sigma_max = alpha[global_param_ids[ParamId::gamma_sigma_max]];
  const double t_shift = alpha[global_param_ids[ParamId::t_shift]];
  const double tau_1 = alpha[global_param_ids[ParamId::tau_1]];
  const double tau_2 = alpha[global_param_ids[ParamId::tau_2]];
  const double m1 = alpha[global_param_ids[ParamId::m1]];
  const double m2 = alpha[global_param_ids[ParamId::m2]];
  const double b_f = alpha[global_param_ids[ParamId::b_f]];
  const double b_t = alpha[global_param_ids[ParamId::b_t]];
  const double Pin = y[global_var_ids[0]];
  const double Qin = y[global_var_ids[1]];
  const double Pout = y[global_var_ids[2]];
  const double Qout = y[global_var_ids[3]];
  const double stress = y[global_var_ids[4]];
  const double tau = y[global_var_ids[5]];
  const double volume = y[global_var_ids[6]];
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
  const double x12 = 1.0/tau_1;
  const double x13 = t - t_shift;
  const double x14 = x12*x13;
  const double x15 = pow(x14, m1);
  const double x16 = x15 + 1;
  const double x17 = 1.0/x16;
  const double x18 = 1.0/tau_2;
  const double x19 = x13*x18;
  const double x20 = pow(x19, m2);
  const double x21 = x20 + 1;
  const double x22 = x15/x21;
  const double x23 = x17*x22;
  const double x24 = volume*x0/x1;
  const double x25 = pow(x2, 5.333333333333333);
  const double x26 = 0.66666666666666663*b_t;
  const double x27 = pow(x2, 2.6666666666666665);
  const double x28 = x3 - 1;
  const double x29 = 1 - x6;
  const double x30 = pow(x2, 3.333333333333333);
  const double x31 = x11*x29 - 0.25*x28*x30*x9;
  const double x32 = x27*x31;
  const double x33 = pow(x29, 2);
  const double x34 = x27*pow(x28, 2);
  const double x35 = 1.0/x27;
  const double x36 = 0.25*x35;
  const double x37 = exp(x36*(b_t*x33 + x34*x9));
  const double x38 = guccione_C*x37;
  const double x39 = pow(x2, -8.0)*x38;
  const double x40 = m2*x20;
  const double x41 = gamma_sigma_max/pow(x16, 2);
  const double x42 = x15/pow(x21, 2);
  const double x43 = x22*x41;
  const double x44 = gamma_sigma_max*x17*x42;

  residual(global_eqn_ids[0]) = stress*x5 - x4;
  residual(global_eqn_ids[1]) = guccione_C*(-b_t*x7*x8 + x10*x3*(0.5*b_f + x11))*exp(b_t*pow(x8, 2) + pow(x10, 2)*x9) - stress + tau;
  residual(global_eqn_ids[2]) = -gamma_sigma_max*x23 + tau;
  residual(global_eqn_ids[3]) = Qin - Qout - dvolume_dt;
  residual(global_eqn_ids[4]) = Pin - Pout;
  jacobian.coeffRef(global_eqn_ids[0], global_param_ids[ParamId::volume0]) = x24*(-0.33333333333333331*stress*x5 + 0.66666666666666663*x4);
  jacobian.coeffRef(global_eqn_ids[1], global_param_ids[ParamId::volume0]) = x24*x39*(x25*(x26*(x6 - 2) + 0.16666666666666666*x27*x9*(-x28*x3 - x6)) + x32*(-x26*x29 + 0.33333333333333331*x28*x30*x9));
  jacobian.coeffRef(global_eqn_ids[1], global_param_ids[ParamId::guccione_C]) = -x31*x35*x37;
  jacobian.coeffRef(global_eqn_ids[2], global_param_ids[ParamId::gamma_sigma_max]) = -x23;
  jacobian.coeffRef(global_eqn_ids[2], global_param_ids[ParamId::t_shift]) = x41*x42*(-m1*x15*x21 + m1*x16*x21 - x16*x40)/x13;
  jacobian.coeffRef(global_eqn_ids[2], global_param_ids[ParamId::tau_1]) = m1*x12*x43;
  jacobian.coeffRef(global_eqn_ids[2], global_param_ids[ParamId::tau_2]) = -x18*x40*x44;
  jacobian.coeffRef(global_eqn_ids[2], global_param_ids[ParamId::m1]) = -x43*log(x14);
  jacobian.coeffRef(global_eqn_ids[2], global_param_ids[ParamId::m2]) = x20*x44*log(x19);
  jacobian.coeffRef(global_eqn_ids[1], global_param_ids[ParamId::b_f]) = x28*x36*x38*(-x28*x31 + x30);
  jacobian.coeffRef(global_eqn_ids[1], global_param_ids[ParamId::b_t]) = x39*(x25*(x30*(0.25*x3 - 0.25) + 0.5*x6 - 0.5) - 0.25*x32*(x33 + x34));
}