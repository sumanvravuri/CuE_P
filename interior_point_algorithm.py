'''
Created on Jul 29, 2013

@author: sumanravuri
'''

import numpy as np
import scipy as sp
import scipy.linalg as sl
import datetime
import re
import logging
import sys

class QP_object:
    def __init__(self):
        self.name = None
        self.quad_obj_mat = None
        self.lin_obj_vec = None
        self.const_obj_scalar = 0.0
        
        self.row_name_to_equality_idx_dict = dict()
        self.row_name_to_inequality_idx_dict = dict()
        self.row_name_to_bound_idx_dict = dict()
        self.col_name_to_idx_dict = dict()
        
        self.equality_constraints_mat = None
        self.equality_constraints_vec = None
        
        self.inequality_constraints_mat = None
        self.inequality_constraints_vec = 0.0
        self.inequality_constraints_operator = list()
        
        self.bound_constraints_dict = dict() #col_name -> (bound, operator)
        self.num_constraints = (0, 0, 0)
        self.num_dim = 0
        
    def qp_obj_to_standard_form(self):
        A = None
        b = None
        
        if self.num_constraints[0] > 0:
            A = np.vstack((self.equality_constraints_mat, -self.equality_constraints_mat))
            b = np.hstack((self.equality_constraints_vec, -self.equality_constraints_vec))
        
        if self.num_constraints[1] > 0:
            for idx, row in enumerate(self.inequality_constraints_mat):
                if self.inequality_constraints_operator[idx] == '>':
                    if A == None:
                        A = row
                        b = self.inequality_constraints_vec[idx]
                        continue
                    A = np.vstack((A, row))
                    b = np.hstack((b, self.inequality_constraints_vec[idx]))
                elif self.inequality_constraints_operator[idx] == '<':
                    if A == None:
                        A = -row
                        b = -self.inequality_constraints_vec[idx]
                        continue
                    A = np.vstack((A, -row))
                    b = np.hstack((b, -self.inequality_constraints_vec[idx]))
        
        if self.num_constraints[2] > 0:
            for bound_col_name in self.bound_constraints_dict:
                col_idx = self.col_name_to_idx_dict[bound_col_name]
                for bound, operator in self.bound_constraints_dict[bound_col_name]:
                    row = np.zeros((self.num_dim,))
                    if operator == 'LO':
                        row[col_idx] = 1.0
                        out_bound = bound
                    else:
                        row[col_idx] = -1.0
                        out_bound = -bound
                    if A == None:
                        A = row
                        b = bound
                        continue
                    A = np.vstack((A, row))
                    b = np.hstack((b, out_bound))
                
        return self.quad_obj_mat, self.lin_obj_vec, A, b

    def qp_obj_to_standard_form_with_equality_constraints(self):
        
        A_inequality = None
        b_inequality = None
        
        A_equality = self.equality_constraints_mat
        b_equality = self.equality_constraints_vec
        
        
        if self.num_constraints[1] > 0:
            for idx, row in enumerate(self.inequality_constraints_mat):
                if self.inequality_constraints_operator[idx] == '>':
                    if A_inequality == None:
                        A_inequality = row
                        b_inequality = self.inequality_constraints_vec[idx]
                    else:
                        A_inequality = np.vstack((A_inequality, row))
                        b_inequality = np.hstack((b_inequality, self.inequality_constraints_vec[idx]))
                elif self.inequality_constraints_operator[idx] == '<':
                    if A_inequality == None:
                        A_inequality = -row
                        b_inequality = -self.inequality_constraints_vec[idx]
                    else:
                        A_inequality = np.vstack((A_inequality, -row))
                        b_inequality = np.hstack((b_inequality, -self.inequality_constraints_vec[idx]))
        
        if self.num_constraints[2] > 0:
            for bound_col_name in self.bound_constraints_dict:
                col_idx = self.col_name_to_idx_dict[bound_col_name]
                for bound, operator in self.bound_constraints_dict[bound_col_name]:
                    row = np.zeros((self.num_dim,))
                    if operator == 'LO':
                        row[col_idx] = 1.0
                        out_bound = bound
                    else:
                        row[col_idx] = -1.0
                        out_bound = -bound
                    if A_inequality == None:
                        A_inequality = row
                        b_inequality = bound
                        continue
                    A_inequality = np.vstack((A_inequality, row))
                    b_inequality = np.hstack((b_inequality, out_bound))
                
        return self.quad_obj_mat, self.lin_obj_vec, A_inequality, b_inequality, A_equality, b_equality

    def read_qps_file(self, file_name):
        print "reading", file_name
        
        fp = open(file_name)
        
        blocks = ['RHS', 'BOUNDS', 'RANGES', 'QUADOBJ']
        space_pattern = re.compile(r' +')
        
        current_block = 'NAME'
        objective_name = 'obj_name'
        
        num_equality_constraints = 0
        num_inequality_constraints = 0
        num_dim = 0
        
        for line in fp:
            cleaned_line = line.strip()
            if cleaned_line == 'ENDATA':
                break
            elif cleaned_line.startswith('*') or cleaned_line == '':
                continue
            cleaned_data = space_pattern.split(cleaned_line)
            if cleaned_data[0] == 'NAME':
                self.name = cleaned_data[1]
                continue
            elif cleaned_data[0] == 'ROWS':
                current_block = 'ROWS'
                continue
            elif cleaned_data[0] == 'COLUMNS':
                current_block = 'COLUMNS'
                continue
            elif cleaned_data[0] in blocks:
                current_block = cleaned_data[0]
                break
            else:
                if current_block == 'ROWS':
                    if cleaned_data[0] == 'N':
                        objective_name = cleaned_data[1]
                        continue
                    elif cleaned_data[0] == 'E':
                        self.row_name_to_equality_idx_dict[cleaned_data[1]] = num_equality_constraints #using as index
                        num_equality_constraints += 1
                        continue
                    elif cleaned_data[0] == 'G' or cleaned_data[0] == 'L':
                        self.row_name_to_inequality_idx_dict[cleaned_data[1]] = (num_inequality_constraints, cleaned_data[0])
                        num_inequality_constraints += 1
                        continue
                elif current_block == 'COLUMNS': #just read column names for now, will deal with actually reading in inequality constraints later
                    if cleaned_data[0] not in self.col_name_to_idx_dict:
                        self.col_name_to_idx_dict[cleaned_data[0]] = num_dim
                        num_dim += 1
                        continue
                else:
                    print "encountered", current_block, "in first three blocks which is not NAME, ROW, or COLUMN... quitting now"
                    quit()
        fp.seek(0)
        
        self.num_dim = num_dim
        #read (in)equality constraints
        if num_equality_constraints > 0:
            self.equality_constraints_mat = np.zeros((num_equality_constraints, num_dim))
        if num_inequality_constraints > 0:
            self.inequality_constraints_mat = np.zeros((num_inequality_constraints, num_dim))
        
        self.quad_obj_mat = np.zeros((num_dim, num_dim))
        self.lin_obj_vec = np.zeros((num_dim,))
        
        for line in fp:
            cleaned_line = line.strip()
            if cleaned_line == 'ENDATA':
                break
            elif cleaned_line.startswith('*') or cleaned_line == '':
                continue
            cleaned_data = space_pattern.split(cleaned_line)
            if cleaned_data[0] == 'NAME':
                self.name = cleaned_data[1]
                continue
            elif cleaned_data[0] == 'ROWS':
                current_block = 'ROWS'
                continue
            elif cleaned_data[0] == 'COLUMNS':
                current_block = 'COLUMNS'
                continue
            elif cleaned_data[0] in blocks:
                current_block = cleaned_data[0]
                break
            if current_block != 'COLUMNS':
                continue
            column_name = cleaned_data[0]
            column_idx = self.col_name_to_idx_dict[column_name]
            row_names = cleaned_data[1::2]
            row_vals = [float(x) for x in cleaned_data[2::2]]
            
            for row_name, row_val in zip(row_names, row_vals):
                if row_name in self.row_name_to_equality_idx_dict:
                    row_idx = self.row_name_to_equality_idx_dict[row_name]
                    self.equality_constraints_mat[row_idx, column_idx] = row_val
                elif row_name in self.row_name_to_inequality_idx_dict:
                    row_idx, operator = self.row_name_to_inequality_idx_dict[row_name]
                    sign = 2 * ((operator == 'G') - 0.5)
                    self.inequality_constraints_mat[row_idx, column_idx] = sign * row_val
                elif row_name == objective_name:
                    self.lin_obj_vec[column_idx] = row_val
        
        num_bound_constraints = 0
        self.equality_constraints_vec = np.zeros((num_equality_constraints,))
        self.inequality_constraints_vec = np.zeros((num_inequality_constraints,))
        
        for line in fp:
            cleaned_line = line.strip()
            
            if cleaned_line.startswith('*') or cleaned_line == '':
                continue
            cleaned_data = space_pattern.split(cleaned_line)
            
            if cleaned_line in blocks:
                current_block = cleaned_data[0]
                continue
            elif cleaned_data[0] == 'ENDATA':
                break
            
            elif current_block == 'RHS':
                row_name = cleaned_data[1]
                row_val = float(cleaned_data[2])
                if row_name in self.row_name_to_equality_idx_dict:
                    row_idx = self.row_name_to_equality_idx_dict[row_name]
                    self.equality_constraints_vec[row_idx] = row_val
                elif row_name in self.row_name_to_inequality_idx_dict:
                    row_idx, operator = self.row_name_to_inequality_idx_dict[row_name]
                    sign = 2 * ((operator == 'G') - 0.5)
                    self.inequality_constraints_vec[row_idx] = sign * row_val
                elif row_name == objective_name:
                    self.const_obj_scalar = row_val
                continue
            elif current_block == 'BOUNDS':
                bound_type = cleaned_data[0]
                column_name = cleaned_data[2]
                bound_val = float(cleaned_data[3])
                if column_name in self.bound_constraints_dict:
                    self.bound_constraints_dict[column_name].append((bound_val, bound_type))
                else:
                    self.bound_constraints_dict[column_name] = [(bound_val, bound_type)]
                num_bound_constraints += 1
            elif current_block == 'QUADOBJ':
                column_name_1 = cleaned_data[0]
                column_name_2 = cleaned_data[1]
                mat_val = float(cleaned_data[2])
                c1 = self.col_name_to_idx_dict[column_name_1]
                c2 = self.col_name_to_idx_dict[column_name_2]
                self.quad_obj_mat[c1, c2] = mat_val
                self.quad_obj_mat[c2, c1] = mat_val
            else:
                current_block, "not recognized... quitting"
                sys.exit(2)
        fp.close()
        self.num_constraints = (num_equality_constraints, num_inequality_constraints, num_bound_constraints)


def calculate_primal_dual_step_size(primal_step_size, dual_step_size, 
                                    residual_lagrangian, residual_inequality, residual_equality, residual_complementary_slackness,
                                    x_step, slacks_step, dual_inequality_step, 
                                    slacks, dual_inequality, second_order_mat, use_min=False):
    if primal_step_size < 0.0:
        print "WARNING, primal_step_size, ", primal_step_size, "is less than zero (means that there is probably a numerical error, setting to 0.0"
        primal_step_size = 0.0
    elif dual_step_size < 0.0:
        print "WARNING, dual_step_size, ", dual_step_size, "is less than zero (means that there is probably a numerical error, setting to 0.0"
        dual_step_size = 0.0
    if primal_step_size == 0.0 and dual_step_size == 0.0:
        print "both primal and dual step sizes are 0.0, meaning no progress will be made... quitting now"
        quit()    
    
    min_step_size = min([primal_step_size, dual_step_size])
    if use_min:
        return min_step_size, min_step_size
    new_lagrangian_residual = np.linalg.norm((1-min_step_size) * residual_lagrangian, ord=float("inf"))
    new_inequality_residual = np.linalg.norm((1-min_step_size) * residual_inequality, ord=float("inf"))
    new_equality_residual = np.linalg.norm((1-min_step_size) * residual_equality, ord=float("inf"))
    new_complementary_slackness_residual = np.linalg.norm((slacks + min_step_size * slacks_step) * 
                                                          (dual_inequality + min_step_size * dual_inequality_step), 
                                                          ord=float("inf"))
    max_equal_step_norm_residual = max((new_lagrangian_residual, new_inequality_residual, new_equality_residual, new_complementary_slackness_residual))
    
    new_lagrangian_residual = np.linalg.norm((1-dual_step_size) * residual_lagrangian 
                                             + (primal_step_size - dual_step_size) * np.dot(second_order_mat, x_step), 
                                             ord=float("inf"))
    new_inequality_residual = np.linalg.norm((1-primal_step_size) * residual_inequality, ord=float("inf"))
    new_equality_residual = np.linalg.norm((1-primal_step_size) * residual_equality, ord=float("inf"))
    new_complementary_slackness_residual = np.linalg.norm((slacks + primal_step_size * slacks_step) * 
                                                          (dual_inequality + dual_step_size * dual_inequality_step), 
                                                          ord=float("inf"))
    max_unequal_step_norm_residual = max((new_lagrangian_residual, new_inequality_residual, new_equality_residual, new_complementary_slackness_residual))
    
    if max_unequal_step_norm_residual < max_equal_step_norm_residual:
        return primal_step_size, dual_step_size
    else:
        return min_step_size, min_step_size

def interior_point_qp_augmented_lagrangian(second_order_mat, first_order_vec, A_equality, b_equality, A_inequality, b_inequality, 
                                           tol=1E-8, max_iterations=100, x_0=None, slack_0=None, dual_inequality_0=None, 
                                           dual_equality_0=None, tau=None, seed=0, verbose = False, use_unequal_primal_dual_steps=True):
    """minimizes function 1/2 x'(second_order_mat)x + (first_order_vec)'x subject to (A_inequality)x >= b, (A_equality)x = b
    using an interior point algorithm
    vu_0 is the dual variable for Ax-y = b condition"""
    start_time = datetime.datetime.now()
    solver_times = list()
    np.random.seed(seed)
    num_equality_constraints = len(b_equality)
    num_inequality_constraints = len(b_inequality)
    num_dims = len(first_order_vec)
    #TODO: better initial parameters
    if x_0 == None:
        x = np.dot(sl.pinv(A_equality), b_equality)
        norm_residual = np.linalg.norm(np.dot(A_equality, x) - b_equality, ord=2)
        if norm_residual > tol:
            print "equality constraints are not satisfiable since residual", norm_residual, "is greater than", tol, "tolerance... quitting"
            sys.exit(3)
    else:
        x = x_0
    if slack_0 == None:
        slacks = np.random.rand(num_inequality_constraints) + 1.
    else:
        slacks = slack_0
    if dual_inequality_0 == None:
        dual_inequality = np.random.rand(num_inequality_constraints) + 1.
    else:
        dual_inequality = dual_inequality_0
    if dual_equality_0 == None:
        dual_equality = np.random.rand(num_equality_constraints) + 1.
    else:
        dual_equality = dual_equality_0
    if tau == None:
        if max_iterations <= 5:
            tau = np.ones((max_iterations,)) * 0.95
        else:
            tau = np.hstack((np.ones((5,)) * 0.5, np.ones(max_iterations-5,) * 0.95))
    zero_mat = np.zeros((num_equality_constraints,num_equality_constraints))
    residual_lagrangian = np.dot(second_order_mat, x) + first_order_vec - np.dot(A_equality.T, dual_equality) - np.dot(A_inequality.T, dual_inequality)
    residual_inequality = np.dot(A_inequality, x) - b_inequality - slacks
    residual_equality = np.dot(A_equality, x) - b_equality
    residual_complementary_slackness = dual_inequality * slacks
    
    residual_augmented_lagrangian = residual_lagrangian + np.dot(A_inequality.T, (residual_complementary_slackness + residual_inequality * dual_inequality)/slacks)
    
    #TRY A REDUCED SYSTEM
    reduced_residual = np.hstack((residual_augmented_lagrangian, residual_equality))
    augmented_second_order_mat = second_order_mat + np.dot(A_inequality.T, A_inequality * (dual_inequality/slacks)[:, np.newaxis])
    
    solver_start_time = datetime.datetime.now()
    affine_step = _schur_complement_solve_symmetric_ul_pd(augmented_second_order_mat, A_equality, zero_mat, 
                                                          -reduced_residual, A_is_diag=False)
    solver_end_time = datetime.datetime.now()
    solver_times.append(solver_end_time - solver_start_time)
    x_affine_step = affine_step[:num_dims]
    dual_equality_affine_step = -affine_step[num_dims:]
    dual_inequality_affine_step = -(dual_inequality + (residual_inequality + np.dot(A_inequality, x_affine_step) * (dual_inequality / slacks)))
    slacks_affine_step = -(residual_complementary_slackness + slacks * dual_inequality_affine_step) / dual_inequality
    x += x_affine_step
    dual_equality += dual_equality_affine_step
    slacks += slacks_affine_step
    slacks = np.abs(slacks)
    slacks[np.where(slacks < 1.0)] = 1.0
    dual_inequality += dual_inequality_affine_step
    dual_inequality = np.abs(dual_inequality)
    dual_inequality[np.where(dual_inequality < 1.0)] = 1.0
    
    is_solved = False
    print "-------------------------------------------------------------------------------------------------"
    print "|iter\t|lagrange res.\t|ineq. res.\t|eq. res.\t|CompSlack res.\t|primal value \t\t|"
    for iteration in range(max_iterations):
        print "-------------------------------------------------------------------------------------------------"
#        print "at iteration", iteration
        if verbose:
            print "minimum of dual inequality is", np.min(dual_inequality)
            print "minimum of slacks is", np.min(slacks)
        #check if KKT conditions satisfied
        residual_lagrangian = np.dot(second_order_mat, x) + first_order_vec - np.dot(A_equality.T, dual_equality) - np.dot(A_inequality.T, dual_inequality)
        residual_inequality = np.dot(A_inequality, x) - b_inequality - slacks
        residual_equality = np.dot(A_equality, x) - b_equality
        residual_complementary_slackness = dual_inequality * slacks
        augmented_second_order_mat = second_order_mat + np.dot(A_inequality.T, A_inequality * (dual_inequality/slacks)[:, np.newaxis])
        inv_augmented_second_order_mat = sl.inv(augmented_second_order_mat)
        #TODO: check if tolerance condition is correct
        max_residual = max([np.linalg.norm(residual_lagrangian, ord=float("inf")), np.linalg.norm(residual_inequality, ord=float("inf")), 
                            np.linalg.norm(residual_equality, ord=float("inf")), np.linalg.norm(residual_complementary_slackness, ord=float("inf"))])
        if max_residual < tol:
            is_solved = True
            break
        
        mu = np.sum(residual_complementary_slackness) / num_inequality_constraints
        residual_augmented_lagrangian = residual_lagrangian + np.dot(A_inequality.T, (residual_complementary_slackness + residual_inequality * dual_inequality)/slacks)
        # update KKT matrix and solve Newton equation for affine step
        reduced_residual = np.hstack((residual_augmented_lagrangian, residual_equality))
        solver_start_time = datetime.datetime.now()
        affine_step = _schur_complement_solve_symmetric_ul_pd(augmented_second_order_mat, A_equality, zero_mat, 
                                                              -reduced_residual, A_is_diag=False, 
                                                              A_inv=inv_augmented_second_order_mat)
        solver_end_time = datetime.datetime.now()
        solver_times.append(solver_end_time - solver_start_time)
        x_affine_step = affine_step[:num_dims]
        dual_inequality_affine_step = -(dual_inequality + (residual_inequality + np.dot(A_inequality, x_affine_step) * (dual_inequality / slacks)))
        slacks_affine_step = -(residual_complementary_slackness + slacks * dual_inequality_affine_step) / dual_inequality
        
        #TODO: get correct affine step size
        neg_indices = np.where(dual_inequality_affine_step < 0.0)
        if neg_indices[0].size > 0:
            dual_inequality_affine_step_size = min(-dual_inequality[neg_indices] / dual_inequality_affine_step[neg_indices])
        else:
            dual_inequality_affine_step_size = 1.0
        
        neg_indices = np.where(slacks_affine_step < 0.0)
        if neg_indices[0].size > 0:
            slacks_affine_step_size = min(-slacks[neg_indices] / slacks_affine_step[neg_indices])
        else:
            slacks_affine_step_size = 1.0
        affine_step_size = max([min([dual_inequality_affine_step_size, slacks_affine_step_size, 1.]), 0.0])
        
        slacks_affine = slacks + affine_step_size * slacks_affine_step
        dual_inequality_affine = dual_inequality + affine_step_size * dual_inequality_affine_step
        mu_affine = np.dot(slacks_affine, dual_inequality_affine) / num_inequality_constraints
        sigma = (mu_affine / mu) ** 3
        #NOT sure if this is the way to handle the problem
        if sigma > 1.0:
            sigma = 1.0
        if verbose: 
            print "sigma is", sigma
            print "mu is", mu
            print "sigma * mu is", sigma * mu
        residual_corrected_complementary_slackness = residual_complementary_slackness + slacks_affine * dual_inequality_affine - sigma * mu
        residual_corrected_augmented_lagrangian = residual_lagrangian + np.dot(A_inequality.T, (residual_corrected_complementary_slackness + residual_inequality * dual_inequality)/slacks)
        reduced_corrected_residual = np.hstack((residual_corrected_augmented_lagrangian, residual_equality))
        solver_start_time = datetime.datetime.now()
        step = _schur_complement_solve_symmetric_ul_pd(augmented_second_order_mat, A_equality, zero_mat, 
                                                       -reduced_corrected_residual, A_is_diag=False,
                                                       A_inv=inv_augmented_second_order_mat)
        solver_end_time = datetime.datetime.now()
        solver_times.append(solver_end_time - solver_start_time)
        #TODO: get correct primal/dual step sizes
        x_step = step[:num_dims]
        dual_equality_step = -step[num_dims:num_dims+num_equality_constraints]
        dual_inequality_step = -(residual_corrected_complementary_slackness / dual_inequality + residual_inequality + np.dot(A_inequality, x_step)) * (dual_inequality / slacks)
        slacks_step = -(residual_corrected_complementary_slackness + slacks * dual_inequality_step) / dual_inequality
        
        neg_indices = np.where(dual_inequality_step < 0.0)
        if neg_indices[0].size > 0:
            dual_inequality_step_size = min(-tau[iteration] * dual_inequality[neg_indices] / dual_inequality_step[neg_indices])
        else:
            dual_inequality_step_size = 1.0
        
        neg_indices = np.where(slacks_step < 0.0)
        if neg_indices[0].size > 0:
            slacks_step_size = min(-tau[iteration] * slacks[neg_indices] / slacks_step[neg_indices])
        else:
            slacks_step_size = 1.0
        if verbose:
            print "dual_inequality_step_size is", dual_inequality_step_size
            print "slacks_step_size is", slacks_step_size
        
        primal_step_size, dual_step_size = calculate_primal_dual_step_size(slacks_step_size, dual_inequality_step_size, 
                                                                           residual_lagrangian, residual_inequality, 
                                                                           residual_equality, residual_complementary_slackness,
                                                                           x_step, slacks_step, dual_inequality_step, 
                                                                           slacks, dual_inequality, second_order_mat, 
                                                                           use_min=(use_unequal_primal_dual_steps == False))
        
        if verbose:
            print "(primal, dual) step size is", (primal_step_size, dual_step_size)
        
        x += x_step * primal_step_size
        slacks += slacks_step * primal_step_size
        dual_inequality += dual_inequality_step * dual_step_size
        dual_equality += dual_equality_step * dual_step_size
        primal_value = 0.5 * np.dot(np.dot(G,x),x) + np.dot(first_order_vec, x)
        print "|%d\t|%.7f\t|%.7f\t|%.7f\t|%.7f\t|%.7f\t|" % (iteration, np.linalg.norm(residual_lagrangian, ord=float("inf")),
                                                             np.linalg.norm(residual_inequality, ord=float("inf")), np.linalg.norm(residual_equality, ord=float("inf")),
                                                             np.linalg.norm(residual_complementary_slackness, ord=float("inf")), primal_value)
    end_time = datetime.datetime.now()
    run_time = end_time - start_time
    print "Solver ran for", run_time
    print "average run_time for each solve call was", sum(solver_times, datetime.timedelta()) / len(solver_times)
    print "with a total time", sum(solver_times, datetime.timedelta()) 
    if is_solved:
        print "solution found, returning parameters"
        print "primal value is", primal_value
        return x
    else:
        print "failed to find solution, returning None"
        return None

def interior_point_qp(second_order_mat, first_order_vec, A_equality, b_equality, A_inequality, b_inequality, 
                      tol=1E-8, max_iterations=100, x_0=None, slack_0=None, dual_inequality_0=None, 
                      dual_equality_0=None, tau=None, seed=0, verbose = False):
    """minimizes function 1/2 x'(second_order_mat)x + (first_order_vec)'x subject to (A_inequality)x >= b, (A_equality)x = b
    using an interior point algorithm
    vu_0 is the dual variable for Ax-y = b condition"""
    start_time = datetime.datetime.now()
    solver_times = list()
    np.random.seed(seed)
    num_equality_constraints = len(b_equality)
    num_inequality_constraints = len(b_inequality)
    num_dims = len(first_order_vec)
    #TODO: better initial parameters
    if x_0 == None:
        x = np.dot(sl.pinv(A_equality), b_equality)
        norm_residual = np.linalg.norm(np.dot(A_equality, x) - b_equality, ord=2)
        if norm_residual > tol:
            print "equality constraints are not satisfiable since residual", norm_residual, "is greater than", tol, "tolerance... quitting"
            sys.exit(3)
#        x = np.zeros(num_dims)
#        x = 5.55 * np.ones(num_dims) #NEED TO REMOVE
    else:
        x = x_0
    if slack_0 == None:
        slacks = np.random.rand(num_inequality_constraints) + 1.
    else:
        slacks = slack_0
    if dual_inequality_0 == None:
        #see "Object-Oriented Software for Quadratic Programming" by E. Gertz and S. Wright for justification
#        max_data_norm = np.max([np.max(np.abs(second_order_mat)), np.max(np.abs(first_order_vec)), np.max(np.abs(A_equality)), 
#                                np.max(np.abs(b_equality)), np.max(np.abs(A_inequality)), np.max(np.abs(b_inequality))])
        dual_inequality = np.random.rand(num_inequality_constraints) + 1.
    else:
        dual_inequality = dual_inequality_0
    if dual_equality_0 == None:
        dual_equality = np.random.rand(num_equality_constraints) + 1.
    else:
        dual_equality = dual_equality_0
    if tau == None:
        if max_iterations <= 5:
            tau = np.ones((max_iterations,)) * 0.95
        else:
            tau = np.hstack((np.ones((5,)) * 0.5, np.ones(max_iterations-5,) * 0.95))
#    print "inequality constraint violations (negative are infeasible) are", np.dot(A_inequality, x) - b_inequality
    #kkt_matrix looks like  [G  A_equality' A_inequality.T]
    #                       [A_equality 0    0 ]
    #                       [A_inequality 0 -S(D_i)^{1} ]
    #where H = G + A_i' diag(slacks_^{-1}) diag(dual_inequality) A_i
    reduced_kkt_matrix = np.zeros((num_dims+num_equality_constraints+num_inequality_constraints, 
                                   num_dims+num_equality_constraints+num_inequality_constraints))
    reduced_kkt_matrix[:num_dims,:num_dims] = G
    reduced_kkt_matrix[:num_dims,num_dims:num_dims+num_equality_constraints] = A_equality.T
    reduced_kkt_matrix[:num_dims,-num_inequality_constraints:] = A_inequality.T
    reduced_kkt_matrix[num_dims:num_dims+num_equality_constraints,:num_dims] = A_equality
    reduced_kkt_matrix[-num_inequality_constraints:,:num_dims] = A_inequality
    reduced_kkt_matrix[-num_inequality_constraints:,-num_inequality_constraints:] = -np.diag((slacks / dual_inequality))
    print "KKT matrix shape is", reduced_kkt_matrix.shape
    residual_lagrangian = np.dot(second_order_mat, x) + first_order_vec - np.dot(A_equality.T, dual_equality) - np.dot(A_inequality.T, dual_inequality)
    residual_inequality = np.dot(A_inequality, x) - b_inequality - slacks
    residual_equality = np.dot(A_equality, x) - b_equality
    residual_complementary_slackness = dual_inequality * slacks
    
    residual_reduced_kkt = residual_inequality + (residual_complementary_slackness / dual_inequality)
    residual = np.hstack((residual_lagrangian, residual_equality, residual_reduced_kkt))
    residual_augmented_lagrangian = residual_lagrangian + np.dot(A_inequality.T, (residual_complementary_slackness + residual_inequality * dual_inequality)/slacks)
    
    #TRY A REDUCED SYSTEM
#    reduced_residual = np.hstack((residual_augmented_lagrangian, residual_equality))
#    augmented_second_order_mat = second_order_mat + np.dot(A_inequality.T, A_inequality * (dual_inequality/slacks)[:, np.newaxis])
    neg_residual = -residual
    
    solver_start_time = datetime.datetime.now()
#    affine_step = sl.solve(reduced_kkt_matrix, neg_residual)
    affine_step = _schur_complement_solve_symmetric(reduced_kkt_matrix[:num_dims+num_equality_constraints,:num_dims+num_equality_constraints], 
                                                    reduced_kkt_matrix[-num_inequality_constraints:,:num_dims+num_equality_constraints], 
                                                    -slacks / dual_inequality, neg_residual, C_is_diag=True)
#    affine_step = _schur_complement_solve_symmetric_ul_pd(augmented_second_order_mat, A_equality, np.zeros((num_equality_constraints,num_equality_constraints)), -reduced_residual, A_is_diag=False)
    solver_end_time = datetime.datetime.now()
    solver_times.append(solver_end_time - solver_start_time)
    x_affine_step = affine_step[:num_dims]
#    print "x_affine_step is"
#    print x_affine_step
    dual_equality_affine_step = -affine_step[num_dims:num_dims+num_equality_constraints]
#    Ax_slacks_affine_step = np.dot(A_inequality, x_affine_step) + residual_inequality
    dual_inequality_affine_step = -affine_step[-num_inequality_constraints:]
    slacks_affine_step = -(residual_complementary_slackness + slacks * dual_inequality_affine_step) / dual_inequality
#    print "slack step computed with np.dot(A_inequality, x_affine_step) + residual_inequality is"
#    print Ax_slacks_affine_step
#    print "slack step computed with -(residual_complementary_slackness - slacks * dual_inequality_affine_step) / dual_inequality is"
#    print slacks_affine_step
    x += x_affine_step
    dual_equality += dual_equality_affine_step
    slacks += slacks_affine_step
    slacks = np.abs(slacks)
    slacks[np.where(slacks < 1.0)] = 1.0
    dual_inequality += dual_inequality_affine_step
    dual_inequality = np.abs(dual_inequality)
    dual_inequality[np.where(dual_inequality < 1.0)] = 1.0
    
    is_solved = False
    print "-------------------------------------------------------------------------------------------------"
    print "|iter\t|lagrange res.\t|ineq. res.\t|eq. res.\t|CompSlack res.\t|primal value \t\t|"
    for iteration in range(max_iterations):
        print "-------------------------------------------------------------------------------------------------"
#        print "at iteration", iteration
        if verbose:
            print "minimum of dual inequality is", np.min(dual_inequality)
            print "minimum of slacks is", np.min(slacks)
        #check if KKT conditions satisfied
        residual_lagrangian = np.dot(second_order_mat, x) + first_order_vec - np.dot(A_equality.T, dual_equality) - np.dot(A_inequality.T, dual_inequality)
        residual_inequality = np.dot(A_inequality, x) - b_inequality - slacks
        residual_equality = np.dot(A_equality, x) - b_equality
        residual_complementary_slackness = dual_inequality * slacks
        
#        print "l_inf norm residual lagrangian is", np.linalg.norm(residual_lagrangian, ord=float("inf"))
#        print "l_inf norm residual inequality condition with slacks is", np.linalg.norm(residual_inequality, ord=float("inf"))
#        print "l_inf norm residual equality condition is", np.linalg.norm(residual_equality, ord=float("inf"))
#        print "l_inf norm residual complementary slackness is", np.linalg.norm(residual_complementary_slackness, ord=float("inf"))
        
        #TODO: check if tolerance condition is correct
        max_residual = max([np.linalg.norm(residual_lagrangian, ord=float("inf")), np.linalg.norm(residual_inequality, ord=float("inf")), 
                            np.linalg.norm(residual_equality, ord=float("inf")), np.linalg.norm(residual_complementary_slackness, ord=float("inf"))])
        if max_residual < tol:
            is_solved = True
            break
        
        mu = np.sum(residual_complementary_slackness) / num_inequality_constraints
        
        residual_reduced_kkt = residual_inequality + (residual_complementary_slackness / dual_inequality)
        residual = np.hstack((residual_lagrangian, residual_equality, residual_reduced_kkt))
        neg_residual = -residual
        # update KKT matrix and solve Newton equation for affine step
        reduced_kkt_matrix[-num_inequality_constraints:,-num_inequality_constraints:] = -np.diag((slacks / dual_inequality))
        
        #affine_step = _schur_complement_solve_symmetric(second_order_mat, A, -y / nu, neg_residual, C_is_diag=True)
        solver_start_time = datetime.datetime.now()
#        affine_step = sl.solve(reduced_kkt_matrix, neg_residual) #linear_conjugate_gradient(reduced_kkt_matrix, neg_residual)
        affine_step = _schur_complement_solve_symmetric(reduced_kkt_matrix[:num_dims+num_equality_constraints,:num_dims+num_equality_constraints], 
                                                        reduced_kkt_matrix[-num_inequality_constraints:,:num_dims+num_equality_constraints], 
                                                        -slacks / dual_inequality, neg_residual, C_is_diag=True)
        solver_end_time = datetime.datetime.now()
        solver_times.append(solver_end_time - solver_start_time)
#        print "l_inf norm of affine step residual is", sl.norm(np.dot(reduced_kkt_matrix, affine_step) - neg_residual, ord=float("inf"))
        x_affine_step = affine_step[:num_dims]
#        dual_equality_affine_step = -affine_step[num_dims:num_dims+num_equality_constraints]
#        Ax_slacks_affine_step = np.dot(A_inequality, x_affine_step) + residual_inequality
        dual_inequality_affine_step = -affine_step[-num_inequality_constraints:]
        slacks_affine_step = -(residual_complementary_slackness + slacks * dual_inequality_affine_step) / dual_inequality
        
#        print "slack step computed with np.dot(A_inequality, x_affine_step) + residual_inequality is"
#        print Ax_slacks_affine_step
#        print "slack step computed with -(residual_complementary_slackness - slacks * dual_inequality_affine_step) / dual_inequality is"
#        print slacks_affine_step
        #TODO: get correct affine step size
        neg_indices = np.where(dual_inequality_affine_step < 0.0)
        if neg_indices[0].size > 0:
            dual_inequality_affine_step_size = min(-dual_inequality[neg_indices] / dual_inequality_affine_step[neg_indices])
        else:
            dual_inequality_affine_step_size = 1.0
        
        neg_indices = np.where(slacks_affine_step < 0.0)
        if neg_indices[0].size > 0:
            slacks_affine_step_size = min(-slacks[neg_indices] / slacks_affine_step[neg_indices])
        else:
            slacks_affine_step_size = 1.0
        affine_step_size = max([min([dual_inequality_affine_step_size, slacks_affine_step_size, 1.]), 0.0])
        
        slacks_affine = slacks + affine_step_size * slacks_affine_step
        dual_inequality_affine = dual_inequality + affine_step_size * dual_inequality_affine_step
        mu_affine = np.dot(slacks_affine, dual_inequality_affine) / num_inequality_constraints
        sigma = (mu_affine / mu) ** 3
        #NOT sure if this is the way to handle the problem
        if sigma > 1.0:
            sigma = 1.0
        if verbose: 
            print "sigma is", sigma
            print "mu is", mu
            print "sigma * mu is", sigma * mu
        residual_corrected_complementary_slackness = residual_complementary_slackness + slacks_affine * dual_inequality_affine - sigma * mu
        residual_corrected_reduced_kkt = residual_inequality + (residual_corrected_complementary_slackness / dual_inequality)
        neg_corrected_residual = -np.hstack((residual_lagrangian, residual_equality, residual_corrected_reduced_kkt))
        solver_start_time = datetime.datetime.now()
#        step = sl.solve(reduced_kkt_matrix, neg_corrected_residual) #linear_conjugate_gradient(reduced_kkt_matrix, neg_corrected_residual)
        step = _schur_complement_solve_symmetric(reduced_kkt_matrix[:num_dims+num_equality_constraints,:num_dims+num_equality_constraints], 
                                                 reduced_kkt_matrix[-num_inequality_constraints:,:num_dims+num_equality_constraints], 
                                                 -slacks / dual_inequality, neg_corrected_residual, C_is_diag=True)
        solver_end_time = datetime.datetime.now()
        solver_times.append(solver_end_time - solver_start_time)
        #step = _schur_complement_solve_symmetric(second_order_mat, A, -y / nu, neg_corrected_residual, C_is_diag=True)
        #TODO: get correct primal/dual step sizes
#        print "l_inf norm of step residual is", sl.norm(np.dot(reduced_kkt_matrix, affine_step) - neg_corrected_residual, ord=float("inf"))
        x_step = step[:num_dims]
        dual_equality_step = -step[num_dims:num_dims+num_equality_constraints]
#        slacks_step = np.dot(A_inequality, x_step) + residual_inequality
        dual_inequality_step = -step[-num_inequality_constraints:]
        slacks_step = -(residual_corrected_complementary_slackness + slacks * dual_inequality_step) / dual_inequality
        
        neg_indices = np.where(dual_inequality_step < 0.0)
        if neg_indices[0].size > 0:
            dual_inequality_step_size = min(-tau[iteration] * dual_inequality[neg_indices] / dual_inequality_step[neg_indices])
        else:
            dual_inequality_step_size = 1.0
        
        neg_indices = np.where(slacks_step < 0.0)
        if neg_indices[0].size > 0:
            slacks_step_size = min(-tau[iteration] * slacks[neg_indices] / slacks_step[neg_indices])
        else:
            slacks_step_size = 1.0
        if verbose:
            print "dual_inequality_step_size is", dual_inequality_step_size
            print "slacks_step_size is", slacks_step_size
        
        primal_step_size, dual_step_size = calculate_primal_dual_step_size(slacks_step_size, dual_inequality_step_size, 
                                                                           residual_lagrangian, residual_inequality, 
                                                                           residual_equality, residual_complementary_slackness,
                                                                           x_step, slacks_step, dual_inequality_step, 
                                                                           slacks, dual_inequality, second_order_mat)
        
        if verbose:
            print "(primal, dual) step size is", (primal_step_size, dual_step_size)
        
        x += x_step * primal_step_size
        slacks += slacks_step * primal_step_size
        dual_inequality += dual_inequality_step * dual_step_size
        dual_equality += dual_equality_step * dual_step_size
        primal_value = 0.5 * np.dot(np.dot(G,x),x) + np.dot(first_order_vec, x)
        print "|%d\t|%.7f\t|%.7f\t|%.7f\t|%.7f\t|%.7f\t|" % (iteration, np.linalg.norm(residual_lagrangian, ord=float("inf")),
                                                             np.linalg.norm(residual_inequality, ord=float("inf")), np.linalg.norm(residual_equality, ord=float("inf")),
                                                             np.linalg.norm(residual_complementary_slackness, ord=float("inf")), primal_value)
#        print "primal value at the end of the iteration is", primal_value
        
#        print "expected l_inf norm for the next iteration is"
#        print "**************************************************"
#        print "l_inf norm residual lagrangian is", np.linalg.norm((1-step_size) * residual_lagrangian, ord=float("inf"))
#        print "l_inf norm residual inequality condition with slacks is", np.linalg.norm((1-step_size) * residual_inequality, ord=float("inf"))
#        print "l_inf norm residual equality condition is", np.linalg.norm((1-step_size) * residual_equality, ord=float("inf"))
#        print "l_inf norm residual complementary slackness is", np.linalg.norm((1-step_size) * residual_complementary_slackness, ord=float("inf"))
#        print "**************************************************"
#        print "slacks are"
#        print slacks
#        print "dual_inequality is"
#        print dual_inequality
    end_time = datetime.datetime.now()
    run_time = end_time - start_time
    print "Solver ran for", run_time
    print "average run_time for each solve call was", sum(solver_times, datetime.timedelta()) / len(solver_times)
    print "with a total time", sum(solver_times, datetime.timedelta()) 
    if is_solved:
        print "solution found, returning parameters"
        print "primal value is", primal_value
        return x
    else:
        print "failed to find solution, returning None"
        return None

def _schur_complement_solve_symmetric_ul_pd(A, B, C, d, A_is_diag=False, A_inv = None):
    """Solves problem:
        [ A B'] z = d
        [ B C ]
        for z. Assumes that A is a symmetric positive definite, matrix"""
    
    if A_inv == None:
        A_inv = sl.inv(A)
        
    A_dim = A.shape[0]
    z_a= d[:A_dim]
    
    if A_is_diag:
        z_b = d[A_dim:] - np.dot(B, z_a / A)
    else:
        z_b = d[A_dim:] - np.dot(B, np.dot(A_inv, z_a)) #sl.solve(A, z_a))
        
    if A_is_diag:
        schur_comp_mat = C - np.dot(B, B.T / A[:,np.newaxis])
    else:
        schur_comp_mat = C - np.dot(np.dot(B, A_inv), B.T)
        
    z_b = sl.solve(schur_comp_mat, z_b)
    
    if A_is_diag:
        z_a /= A
    else:
        z_a = np.dot(A_inv, z_a) #sl.solve(A, z_a, sym_pos=True, overwrite_b=True)
    #z_b finished, compute z_a
    
    if A_is_diag: #BUGS HERE
        z_a -= np.dot(B.T, z_b) / A
    else:
        z_a -= np.dot(A_inv, np.dot(B.T, z_b)) #sl.solve(A, np.dot(B.T, z_b))
        
    return np.reshape(np.hstack((z_a, z_b)), d.shape)

def _schur_complement_solve_symmetric(A, B, C, d, C_is_diag=False, C_inv = None):
    """Solves problem:
        [ A B'] z = d
        [ B C ]
        for z. Assumes that C is a symmetric positive definite, matrix"""
    A_dim = A.shape[0]
    z_b = d[A_dim:]
    if C_inv == None:
        C_inv = sl.inv(C)
    
    if C_is_diag:
        z_a = d[:A_dim] - np.dot(B.T, z_b / C)
    else:
#        z_a = d[:A_dim] - np.dot(B.T, np.dot(sl.inv(C), z_b))
        z_a = d[:A_dim] - np.dot(B.T, np.dot(C_inv, z_b))#sl.solve(C, z_b, sym_pos=True))
    #SOLVED UPPER TRIANGULAR MATRIX, now middle one
    
    if C_is_diag:
        schur_comp_mat = A - np.dot(B.T,B / C[:, np.newaxis])
    else:
        schur_comp_mat = A - np.dot(np.dot(B.T, C_inv), B) # A - B' C^{-1} B
    
    z_a = sl.solve(schur_comp_mat , z_a)#, sym_pos=True, overwrite_b=True)
    
    if C_is_diag:
        z_b /= C
    else:
        z_b = np.dot(C_inv, z_b) #sl.solve(C, z_b, sym_pos=True, overwrite_b=True)
        
    #SOLVED MIDDLE MATRIX, now lower triangular, z_a already finished
    if C_is_diag:
        z_b -= np.dot(B, z_a) / C
    else:
        z_b -= np.dot(C_inv, np.dot(B, z_a), sym_pos=True) #sl.solve(C, np.dot(B, z_a), sym_pos=True) 
    
    return np.reshape(np.hstack((z_a, z_b)), d.shape)

def linear_conjugate_gradient(second_order_mat, first_order_vec, num_epochs = None, damping_factor = 0.0, #seems to be correct, compare with conjugate_gradient.py
                              verbose = False, x_0 = None, residual_norm_condition=1E-5, function_decrease_condition=1E-7,
                              preconditioner = None):
    """minimizes function f(x) = -b'x + 1/2 * x'(G + d)x using linear conjugate gradient
    where d is the damping factor
    G is the second order matrix
    b is a first order vector"""
    n_dim = second_order_mat.shape[1]
    if verbose:
        print "preconditioner is", preconditioner
    if num_epochs == None:
        num_epochs = n_dim
    if x_0 == None:
        x_0 = np.random.randn(n_dim, 1)
    if damping_factor < 0.0:
        print "WARNING, damping_factor", damping_factor, "should be >= 0.0"
    if len(first_order_vec.shape) == 1:
        first_order_vec = np.reshape(first_order_vec, (first_order_vec.size, 1))
        
    x = x_0
    Gx = np.dot(second_order_mat, x)
    if damping_factor != 0.0:
        Gx += x * damping_factor
    residual = Gx - first_order_vec
    current_function_val = 0.5 * np.vdot(x, residual - first_order_vec)
    
    if preconditioner != None:
        preconditioned_residual = residual / preconditioner
    else:
        preconditioned_residual = residual
    search_direction = -preconditioned_residual
    residual_dot = np.vdot(preconditioned_residual, residual)
    
    for epoch in range(num_epochs):
#        print "\r                                                                \r", #clear line
        sys.stdout.write("conjugate gradient epoch %d of %d\n" % (epoch+1, num_epochs)), sys.stdout.flush()
        second_order_direction = np.dot(second_order_mat, search_direction)
        if damping_factor != 0.0:
            second_order_direction += search_direction * damping_factor
                        
        curvature = np.vdot(second_order_direction, search_direction)
        if curvature <= 0:
            print "curvature must be positive, but is instead", curvature, "... quitting now!"
            return x
            #sys.exit(2)
        
        step_size = residual_dot / curvature
        if verbose:
            print "residual dot search direction is", np.vdot(search_direction, residual)
            print "residual dot is", residual_dot
            print "curvature is", curvature
            print "step size is", step_size
        x += search_direction * step_size
        
        residual += second_order_direction * step_size
        prev_function_val = current_function_val
        current_function_val = 0.5 * np.vdot(residual - first_order_vec, x)
        if verbose:
            print "model val at end of epoch is", current_function_val
        if (prev_function_val - current_function_val) < function_decrease_condition or sl.norm(residual, ord=2) < residual_norm_condition: #checking termination condition
            if verbose:
                print "\r                                                                \r", #clear line
                sys.stdout.write("termination condition satisfied at iteration %d of %d for conjugate gradient, returning step\n" % (epoch+1, num_epochs)), sys.stdout.flush()
            break
        if preconditioner != None:
            preconditioned_residual = residual / preconditioner
        else:
            preconditioned_residual = residual
        new_residual_dot = np.vdot(preconditioned_residual, residual)
        conjugate_gradient_const = new_residual_dot / residual_dot
        search_direction = -preconditioned_residual + search_direction * conjugate_gradient_const
        residual_dot = new_residual_dot
    return np.reshape(x, (x.size,))


if __name__ == '__main__':
    print "testing interior point algorithm"
    qp_obj = QP_object()
    qp_obj.read_qps_file('qps_files/CVXQP3_M.SIF.txt')
    G,c,A_inequality, b_inequality, A_equality, b_equality = qp_obj.qp_obj_to_standard_form_with_equality_constraints()
    x = interior_point_qp_augmented_lagrangian(G, c, A_equality, b_equality, A_inequality, b_inequality, seed=0, tol=1E-5)
    