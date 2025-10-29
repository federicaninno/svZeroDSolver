// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause
#include "LevenbergMarquardtOptimizer.h"

#include <iomanip>

LevenbergMarquardtOptimizer::LevenbergMarquardtOptimizer(
    Model* model, int num_obs, int num_params, double lambda0, double tol_grad,
    double tol_inc, int max_iter) {
  this->model = model;
  this->num_obs = num_obs;
  this->num_params = num_params;
  this->num_eqns = model->dofhandler.get_num_equations();
  this->num_vars = model->dofhandler.get_num_variables();
  this->num_dpoints = this->num_obs * this->num_eqns;
  this->lambda = lambda0;
  this->tol_grad = tol_grad;
  this->tol_inc = tol_inc;
  this->max_iter = max_iter;

  jacobian = Eigen::SparseMatrix<double>(num_dpoints, num_params);
  residual = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(num_dpoints);
  mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(num_params,
                                                              num_params);
  vec = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(num_params);
}

Eigen::Matrix<double, Eigen::Dynamic, 1> LevenbergMarquardtOptimizer::run(
    Eigen::Matrix<double, Eigen::Dynamic, 1> alpha,
    std::vector<std::vector<double>>& y_obs,
    std::vector<std::vector<double>>& dy_obs) {

  for (size_t i = 0; i < max_iter; ++i) {

    // Print current parameters (alpha) 
    std::cout << "Iteration " << i << " alpha = [";
    for (int idx = 0; idx < alpha.size(); ++idx) {
      std::cout << alpha[idx];
      if (idx < alpha.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Build J and r at current alpha
    update_gradient(alpha, y_obs, dy_obs);

    // Cost evaluation 
    double cost = 0.5 * residual.squaredNorm();

    // Backup J and r
    Eigen::SparseMatrix<double> jac_backup = jacobian;
    Eigen::Matrix<double, Eigen::Dynamic, 1> resid_backup = residual;

    bool accepted = false;
    int attempts = 0;
    const int max_attempts = 12;        // try a few increases of lambda
    const double lambda_increase = 10; // factor to increase lambda on rejection
    const double lambda_decrease = 0.1; // factor to reduce lambda on success

    // Try to find an acceptable step by adjusting lambda
    while (!accepted && attempts < max_attempts) {
      // compute delta for the current lambda
      update_delta(false);

      // trial parameters (note: (alpha - delta) as original code)
      Eigen::Matrix<double, Eigen::Dynamic, 1> alpha_trial = alpha - delta;

      // Evaluate trial cost to deco
      update_gradient(alpha_trial, y_obs, dy_obs);
      double cost_trial = 0.5 * residual.squaredNorm();

      std::cout << std::setprecision(1) << std::scientific
                << "Iter " << i << " attempt " << attempts
                << " lambda " << lambda << " cost " << cost
                << " cost_trial " << cost_trial << std::endl;

      if (cost_trial < cost) {
        // Accept the step
        accepted = true;
        // accept the trial alpha
        alpha = alpha_trial;
        // reduce lambda to be less damped (more Gauss-Newton like)
        lambda = std::max(lambda * lambda_decrease, 1e-16);
        // J and r are already set to those of alpha_trial, so continue
      } else {
        // Reject the step: restore jacobian/residual and increase lambda
        jacobian = jac_backup;
        residual = resid_backup;
        lambda *= lambda_increase;
        ++attempts;
      }
    } // end attempts

    if (!accepted) {
      std::cout << "Warning: no acceptable step found in " << max_attempts
                << " attempts (lambda now " << lambda << "). Stopping.\n";
      break; // or continue; usually better to stop or use gradient step
    }

    // Recompute gradient vector and delta for accepted alpha (for diagnostics)
    update_gradient(alpha, y_obs, dy_obs);
    vec = jacobian.transpose() * residual;
    double norm_grad = vec.norm();
    double norm_inc = delta.norm();

    std::cout << std::setprecision(1) << std::scientific << "Iteration " << i + 1 << " | lambda: " << lambda
              << " | norm inc: " << norm_inc << " | norm grad: " << norm_grad
              << std::endl;

    // Stopping conditions
    if ((norm_grad < tol_grad) && (norm_inc < tol_inc)) {
      std::cout << "Converged: gradient or increment below tolerance.\n";
      break;
    }
  } // end iterations

  return alpha;
}


void LevenbergMarquardtOptimizer::update_gradient(
    Eigen::Matrix<double, Eigen::Dynamic, 1>& alpha,
    std::vector<std::vector<double>>& y_obs,
    std::vector<std::vector<double>>& dy_obs) {
  // Set jacobian and residual to zero
  jacobian.setZero();
  residual.setZero();

  // Build per-row reference magnitudes r_ref[row] from observations
  // Compute a reference scale for each residual row (row = obs_i * num_eqns +
  // eq) r_ref[row] = max(1.0, max_abs of relevant observed y/dy for that
  // block/time)
  std::vector<double> r_ref(num_dpoints, 1.0);

  // For each observation time and each block, look at the block's
  // global_var_ids to obtain representative observed magnitudes for the
  // equations that block contributes.
  for (size_t i = 0; i < (size_t)num_obs; ++i) {
    for (size_t j = 0; j < model->get_num_blocks(true); ++j) {
      Block* block = model->get_block(j);
      // For each equation row this block contributes (local index l)
      for (size_t l = 0; l < block->global_eqn_ids.size(); ++l) {
        // compute absolute row index in global residual vector for observation
        // i
        int base_row = block->global_eqn_ids[l];  // local eq id (0..num_eqns-1)
        int row = base_row +
                  num_eqns * static_cast<int>(i);  // absolute row in residual

        // find representative magnitude from observed variables the block uses
        double local_max = 0.0;
        for (int vid : block->global_var_ids) {
          // global variable index vid; check bounds
          if ((size_t)vid < y_obs[i].size()) {
            local_max = std::max(local_max, std::abs(y_obs[i][vid]));
          }
          if ((size_t)vid < dy_obs[i].size()) {
            local_max = std::max(local_max, std::abs(dy_obs[i][vid]));
          }
        }

        // If we found nothing useful (local_max == 0), leave at 1.0; otherwise
        // use it. Also guard from extremely small refs:
        const double MIN_REF =
            1e-12;  // should avoid division by zero, but not overweight small residuals
        r_ref[row] = std::max(MIN_REF, local_max);
      }
    }
  }

  // Precompute inverse reference (scaling factors to multiply by)
  Eigen::VectorXd inv_ref(num_dpoints);
  for (int r = 0; r < num_dpoints; ++r) inv_ref[r] = 1.0 / r_ref[r];

  // Assemble gradient and residual
  for (size_t i = 0; i < num_obs; i++) {
    for (size_t j = 0; j < model->get_num_blocks(true); j++) {
      auto block = model->get_block(j);
      // Temporarily offset equation indices for this observation
      for (size_t l = 0; l < block->global_eqn_ids.size(); l++) {
        //block->global_eqn_ids[l] += num_eqns * i;
        block->global_eqn_ids[l] += num_eqns * static_cast<int>(i);
      }
      block->update_gradient(jacobian, residual, alpha, y_obs[i], dy_obs[i]);
      for (size_t l = 0; l < block->global_eqn_ids.size(); l++) {
        //block->global_eqn_ids[l] -= num_eqns * i;
        block->global_eqn_ids[l] -= num_eqns * static_cast<int>(i);
      }
    }
  }

  // Normalize residual rows and Jacobian rows by inv_ref
  // Scale residual vector
  for (int r = 0; r < num_dpoints; ++r) {
    residual[r] *= inv_ref[r];
  }

  // Scale sparse jacobian rows: multiply each nonzero by inv_ref[row]
  for (int k = 0; k < jacobian.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(jacobian, k); it; ++it) {
      it.valueRef() *= inv_ref[it.row()];
    }
  }

  // Now jacobian and residual are normalized per-row and ready for building J^T
  // r, J^T J, etc.
}

void LevenbergMarquardtOptimizer::update_delta(bool /*first_step*/) {
  // compute gradient vector (g = J^T * r)
  vec = jacobian.transpose() * residual;

  // compute J^T J and diagonal
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> jacobian_sq =
     jacobian.transpose() * jacobian;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> jacobian_sq_diag =
     jacobian_sq.diagonal().asDiagonal();

  // build LM matrix (note: lambda multiplies the diagonal)
  mat = jacobian_sq + lambda * jacobian_sq_diag;

  // Solve for delta
  delta = mat.llt().solve(vec);
}
