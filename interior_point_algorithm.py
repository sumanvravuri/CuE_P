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
import copy
import sys
from scipy.linalg import fblas as FB

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
        self.inequality_constraints_vec = None
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
        
        A_inequality = self.inequality_constraints_mat
        b_inequality = self.inequality_constraints_vec
        
        A_equality = self.equality_constraints_mat
        b_equality = self.equality_constraints_vec
        
        if self.num_constraints[1] > 0:
            A_inequality[np.where([ x == '<' for x in self.inequality_constraints_operator])] *= -1
            b_inequality[np.where([ x == '<' for x in self.inequality_constraints_operator])] *= -1
        
        if self.num_constraints[2] > 0:
            bound_indices = [(self.col_name_to_idx_dict[col_name], (1 - 2*('UP' in bound)) * bound[0])
                             for col_name in self.bound_constraints_dict for bound in self.bound_constraints_dict[col_name]]
            bound_indices.sort()
            A_bounds = np.zeros((len(bound_indices), self.num_dim))
            A_bounds[(range(len(bound_indices)), [x for x, bound in bound_indices])] = 1.0
            b_bounds = np.array([bound for x, bound in bound_indices])
            if A_inequality == None:
                A_inequality = A_bounds
                b_inequality = b_bounds
            else:
                A_inequality = np.vstack((A_inequality, A_bounds))
                b_inequality = np.hstack((b_inequality, b_bounds))
        #THIS IS BETTER IF I HAD A SPARSE SOLVER
#        if self.num_constraints[2] > 0:
#            lower_bound_indices = [(self.col_name_to_idx_dict[col_name], bound[0]) for col_name in self.bound_constraints_dict for bound in self.bound_constraints_dict[col_name] if 'LO' in bound]
#            
#            if len(lower_bound_indices) > 0:
#                lower_bound_indices.sort()
#                A_lower_bound = np.zeros((len(lower_bound_indices), self.num_dim))
#                A_lower_bound[(range(len(lower_bound_indices)), [x for x, bound in lower_bound_indices])] = 1.0
#                b_lower_bound = np.array([bound for x, bound in lower_bound_indices])
#                if A_inequality == None:
#                    A_inequality = A_lower_bound
#                    b_inequality = b_lower_bound
#                else:
#                    A_inequality = np.vstack((A_inequality, A_lower_bound))
#                    b_inequality = np.hstack((b_inequality, b_lower_bound))
#            
#            upper_bound_indices = [(self.col_name_to_idx_dict[col_name], bound[0]) for col_name in qp_obj.bound_constraints_dict for bound in self.bound_constraints_dict[col_name] if 'UP' in bound]
#            if len(lower_bound_indices) > 0:
#                upper_bound_indices.sort()
#                A_upper_bound = np.zeros((len(upper_bound_indices), self.num_dim))
#                A_upper_bound[(range(len(upper_bound_indices)), [x for x, bound in upper_bound_indices])] = -1.0
#                b_upper_bound = np.array([-bound for x, bound in upper_bound_indices])
#                if A_inequality == None:
#                    A_inequality = A_upper_bound
#                    b_inequality = b_upper_bound
#                else:
#                    A_inequality = np.vstack((A_inequality, A_upper_bound))
#                    b_inequality = np.hstack((b_inequality, b_upper_bound))


                #NAIVE IMPLEMENTATION
#        if self.num_constraints[2] > 0:
#            for bound_col_name in self.bound_constraints_dict:
#                col_idx = self.col_name_to_idx_dict[bound_col_name]
#                for bound, operator in self.bound_constraints_dict[bound_col_name]:
#                    row = np.zeros((self.num_dim,))
#                    if operator == 'LO':
#                        row[col_idx] = 1.0
#                        out_bound = bound
#                    elif operator == 'UP':
#                        row[col_idx] = -1.0
#                        out_bound = -bound
#                    else:
#                        print operator, "not understood... should be either 'LO' or 'UP'"
#                        sys.exit(-1)
#                    if A_inequality == None:
#                        A_inequality = row
#                        b_inequality = bound
#                        continue
#                    A_inequality = np.vstack((A_inequality, row))
#                    b_inequality = np.hstack((b_inequality, out_bound))
                
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


def _calculate_primal_dual_step_size(primal_step_size, dual_step_size, 
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
    
def interior_point_qp_augmented_lagrangian_multi_corrector(second_order_mat, first_order_vec, A_equality, b_equality, A_inequality, b_inequality, 
                                                           tol=1E-8, max_iterations=100, x_0=None, slack_0=None, dual_inequality_0=None, 
                                                           dual_equality_0=None, tau=None, seed=0, verbose = False, use_unequal_primal_dual_steps=False,
                                                           beta_min=0.1, beta_max=10.0, gamma=0.1, delta=0.1, max_num_correctors=5):
    """minimizes function 1/2 x'(second_order_mat)x + (first_order_vec)'x subject to (A_inequality)x >= b, (A_equality)x = b
    using an interior point algorithm
    vu_0 is the dual variable for Ax-y = b condition"""
    start_time = datetime.datetime.now()
    solver_times = list()
    augmented_mat_creation_times = list()
    np.random.seed(seed)
    num_equality_constraints = len(b_equality)
    num_inequality_constraints = len(b_inequality)
    num_dims = len(first_order_vec)
    #TODO: better initial parameters
    if x_0 == None:
        x, residual, rank, singular_values = np.linalg.lstsq(A_equality, b_equality)
        norm_residual = np.linalg.norm(residual, ord=2)
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
        dual_equality = np.random.randn(num_equality_constraints)
    else:
        dual_equality = dual_equality_0
    if tau == None:
        if max_iterations <= 5:
            tau = np.ones((max_iterations,)) * 0.95
        else:
            tau = np.hstack((np.ones((5,)) * 0.5, np.ones(max_iterations-5,) * 0.95))
    zero_mat = np.zeros((num_equality_constraints,num_equality_constraints))
    residual_lagrangian, residual_inequality, residual_equality, residual_complementary_slackness = _calculate_residuals(x, second_order_mat, first_order_vec, A_inequality, A_equality, slacks, dual_inequality, dual_equality)
    
    residual_augmented_lagrangian = residual_lagrangian + np.dot(A_inequality.T, (residual_complementary_slackness + residual_inequality * dual_inequality)/slacks)
    
    #TRY A REDUCED SYSTEM
    reduced_residual = np.hstack((residual_augmented_lagrangian, residual_equality))
    creation_start_time = datetime.datetime.now()
#    augmented_second_order_mat = second_order_mat + np.dot(A_inequality.T, A_inequality * (dual_inequality/slacks)[:, np.newaxis])
    A_half = A_inequality * np.sqrt(dual_inequality/slacks)[:, np.newaxis]
    augmented_second_order_mat = second_order_mat +  FB.dgemm(alpha=1.0, a=A_half.T, b=A_half.T, trans_b=True)
    creation_end_time = datetime.datetime.now()
    augmented_mat_creation_times.append(creation_end_time-creation_start_time)
    
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
    dual_inequality += dual_inequality_affine_step
    violation = max([np.max(-dual_inequality), np.max(-slacks), 0.0])
    shift = 1000 + 2 * violation
    slacks += shift
    dual_inequality += shift
    
    is_solved = False
    print "----------------------------------------------------------------------------------------------------------------------------------------"
    print "|iter\t|#corr.\t|lagrange res.\t|ineq. res.\t|eq. res.\t|CompSlack res.\t|primal value \t\t|mu \t\t|sigma \t\t|"
    for iteration in range(max_iterations):
        print "----------------------------------------------------------------------------------------------------------------------------------------"

        #check if KKT conditions satisfied
        residual_lagrangian, residual_inequality, residual_equality, residual_complementary_slackness = _calculate_residuals(x, second_order_mat, first_order_vec, A_inequality, A_equality, slacks, dual_inequality, dual_equality)
        max_residual = max([np.linalg.norm(residual_lagrangian, ord=float("inf")), np.linalg.norm(residual_inequality, ord=float("inf")), 
                            np.linalg.norm(residual_equality, ord=float("inf")), np.linalg.norm(residual_complementary_slackness, ord=float("inf"))])
        if max_residual < tol:
            is_solved = True
            break
        
        #RUN PREDICTOR STEP
        creation_start_time = datetime.datetime.now()
        A_half = A_inequality * np.sqrt(dual_inequality/slacks)[:, np.newaxis]
        augmented_second_order_mat = second_order_mat +  FB.dgemm(alpha=1.0, a=A_half.T, b=A_half.T, trans_b=True) #        augmented_second_order_mat = second_order_mat + np.dot(A_inequality.T, A_inequality * (dual_inequality/slacks)[:, np.newaxis])
        creation_end_time = datetime.datetime.now()
        augmented_mat_creation_times.append(creation_end_time-creation_start_time)
        inv_augmented_second_order_mat = sl.inv(augmented_second_order_mat)
        neg_schur_part = FB.dgemm(alpha=1.0, a=np.dot(A_equality, inv_augmented_second_order_mat).T, b=A_equality.T, trans_a=True)
        try:
            neg_schur_part_chol_factor = sl.cho_factor(neg_schur_part)
        except np.linalg.linalg.LinAlgError:
            neg_schur_part_chol_factor = None
        #TODO: check if tolerance condition is correct
        mu = np.sum(residual_complementary_slackness) / num_inequality_constraints
        
        # update KKT matrix and solve Newton equation for affine step
        residual_augmented_lagrangian = residual_lagrangian + np.dot(A_inequality.T, (residual_complementary_slackness + residual_inequality * dual_inequality)/slacks)
        reduced_residual = np.hstack((residual_augmented_lagrangian, residual_equality))
        solver_start_time = datetime.datetime.now()
        affine_step = _schur_complement_solve_symmetric_ul_pd(augmented_second_order_mat, A_equality, zero_mat, 
                                                              -reduced_residual, A_is_diag=False, 
                                                              B_Ainv_BT = neg_schur_part, A_inv=inv_augmented_second_order_mat,
                                                              neg_schur_comp_chol_factor = neg_schur_part_chol_factor)
        solver_end_time = datetime.datetime.now()
        solver_times.append(solver_end_time - solver_start_time)
        x_affine_step = affine_step[:num_dims]
        dual_inequality_affine_step = -(dual_inequality + (residual_inequality + np.dot(A_inequality, x_affine_step) * (dual_inequality / slacks)))
        dual_equality_affine_step = -affine_step[num_dims:]
        slacks_affine_step = -(residual_complementary_slackness + slacks * dual_inequality_affine_step) / dual_inequality
        
        #get primal and dual step sizes
        
        dual_inequality_affine_step_size = _max_val_pos_step_size(dual_inequality_affine_step, dual_inequality)
        slacks_affine_step_size = _max_val_pos_step_size(slacks_affine_step, slacks)
        affine_step_size = min(dual_inequality_affine_step_size, slacks_affine_step_size)
        
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
            
        #RUN FIRST-ORDER PREDICTOR/CORRECTOR STEP
        residual_corrected_complementary_slackness = residual_complementary_slackness + slacks_affine_step * dual_inequality_affine_step - sigma * mu
        residual_corrected_augmented_lagrangian = residual_lagrangian + np.dot(A_inequality.T, (residual_corrected_complementary_slackness + residual_inequality * dual_inequality)/slacks)
        reduced_corrected_residual = np.hstack((residual_corrected_augmented_lagrangian, residual_equality))
        solver_start_time = datetime.datetime.now()
        step = _schur_complement_solve_symmetric_ul_pd(augmented_second_order_mat, A_equality, zero_mat, 
                                                       -reduced_corrected_residual, A_is_diag=False,
                                                       B_Ainv_BT = neg_schur_part, A_inv=inv_augmented_second_order_mat,
                                                       neg_schur_comp_chol_factor = neg_schur_part_chol_factor)
        solver_end_time = datetime.datetime.now()
        solver_times.append(solver_end_time - solver_start_time)
        #TODO: get correct primal/dual step sizes
        x_step = step[:num_dims]
        dual_equality_step = -step[num_dims:num_dims+num_equality_constraints]
        dual_inequality_step = -(residual_corrected_complementary_slackness / dual_inequality + residual_inequality + np.dot(A_inequality, x_step)) * (dual_inequality / slacks)
        slacks_step = -(residual_corrected_complementary_slackness + slacks * dual_inequality_step) / dual_inequality
        
        dual_inequality_step_size = _max_val_pos_step_size(dual_inequality_step, dual_inequality, scaling_parameter=tau[iteration])
        slacks_step_size = _max_val_pos_step_size(slacks_step, slacks, scaling_parameter=tau[iteration])
        
        predictor_corrector_step_size = min(slacks_step_size, dual_inequality_step_size)
        #RUN HIGHER-ORDER PREDICTOR/CORRECTOR STEPS
        num_correctors = 0
        for corrector_iter in range(max_num_correctors):
#            higher_order_complementary_slackness = (slacks + slacks_step_size * slacks_step) * (dual_inequality + dual_inequality_step_size * dual_inequality_step)
            StepFactor0 = 0.08 
            StepFactor1 = 1.08
            AcceptTol = 0.005
            target_step_size = min(StepFactor1 * predictor_corrector_step_size + StepFactor0, 1.0);
#            print (np.min(residual_complementary_slackness), np.max(residual_complementary_slackness))
#            print (predictor_corrector_step_size, target_step_size)
            higher_order_complementary_slackness = (slacks + target_step_size * slacks_step) * (dual_inequality + target_step_size * dual_inequality_step)
            min_val = beta_min * sigma * mu
            max_val = beta_max * sigma * mu
#            print (min_val, max_val)
#            print (np.min(higher_order_complementary_slackness), np.max(higher_order_complementary_slackness))
            higher_order_corrector = _gondzio_projection(higher_order_complementary_slackness, min_val, max_val)
#            print (np.min(higher_order_corrector), np.max(higher_order_corrector))
            
            residual_higher_order_complementary_slackness =  higher_order_corrector
            residual_higher_order_augmented_lagrangian =  np.dot(A_inequality.T, residual_higher_order_complementary_slackness/slacks)
            reduced_higher_order_residual = np.hstack((residual_higher_order_augmented_lagrangian, np.zeros(residual_equality.shape)))
            solver_start_time = datetime.datetime.now()
            higher_order_step = _schur_complement_solve_symmetric_ul_pd(augmented_second_order_mat, A_equality, zero_mat, 
                                                                        reduced_higher_order_residual, A_is_diag=False,
                                                                        B_Ainv_BT = neg_schur_part, A_inv=inv_augmented_second_order_mat,
                                                                        neg_schur_comp_chol_factor = neg_schur_part_chol_factor)
            solver_end_time = datetime.datetime.now()
            solver_times.append(solver_end_time - solver_start_time)
            #TODO: get correct primal/dual step sizes
            x_higher_order_step = higher_order_step[:num_dims]
            dual_equality_higher_order_step = -higher_order_step[num_dims:num_dims+num_equality_constraints]
            dual_inequality_higher_order_step = -(residual_higher_order_complementary_slackness / dual_inequality + np.dot(A_inequality, x_higher_order_step)) * (dual_inequality / slacks)
            slacks_higher_order_step = -(residual_higher_order_complementary_slackness + slacks * dual_inequality_higher_order_step) / dual_inequality
            
            x_higher_order_step += x_step
            dual_equality_higher_order_step += dual_equality_step
            dual_inequality_higher_order_step += dual_inequality_step
            slacks_higher_order_step += slacks_step
            
            dual_inequality_higher_order_step_size = _max_val_pos_step_size(dual_inequality_step + dual_inequality_higher_order_step, dual_inequality)
            slacks_higher_order_step_size = _max_val_pos_step_size(slacks_step + slacks_higher_order_step, slacks)
            higher_order_step_size = min(dual_inequality_higher_order_step_size, slacks_higher_order_step_size)
#            print (predictor_corrector_step_size, higher_order_step_size)
             
            if higher_order_step_size >= (1.0+AcceptTol) * predictor_corrector_step_size:
#            dual_inequality_higher_order_step_size > dual_inequality_step_size + gamma * delta and slacks_higher_order_step_size > slacks_step_size + gamma * delta:
                x_step += x_higher_order_step
                dual_equality_step += dual_equality_higher_order_step
                dual_inequality_step += dual_inequality_higher_order_step
                slacks_step += slacks_higher_order_step
                slacks_step_size = slacks_higher_order_step_size
                dual_inequality_step_size = dual_inequality_higher_order_step_size
                predictor_corrector_step_size = min(slacks_step_size, dual_inequality_step_size)
#                residual_corrected_complementary_slackness = residual_higher_order_complementary_slackness
                num_correctors += 1
                if higher_order_step_size == 1.0:
                    break
                continue
            else:
#                print "Used", corrector_iter, "correctors"
                break
        
#        dual_inequality_step_size = _max_val_pos_step_size(dual_inequality_step, dual_inequality, scaling_parameter=tau[iteration])
#        slacks_step_size = _max_val_pos_step_size(slacks_step, slacks, scaling_parameter=tau[iteration])
        slacks_step_size, dual_inequality_step_size = _calculate_mehrotra_step_sizes(slacks, slacks_step, dual_inequality, dual_inequality_step)
        if verbose:
            print "dual_inequality_step_size is", dual_inequality_step_size
            print "slacks_step_size is", slacks_step_size
        
        primal_step_size, dual_step_size = _calculate_primal_dual_step_size(slacks_step_size, dual_inequality_step_size, 
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
        primal_value = 0.5 * np.dot(np.dot(second_order_mat,x),x) + np.dot(first_order_vec, x)
        print "|%d\t|%d\t|%.7f\t|%.7f\t|%.7f\t|%.7f\t|%.7f\t|%.7f\t|%.7f\t|" % (iteration, num_correctors, np.linalg.norm(residual_lagrangian, ord=float("inf")),
                                                                               np.linalg.norm(residual_inequality, ord=float("inf")), np.linalg.norm(residual_equality, ord=float("inf")),
                                                                               np.linalg.norm(residual_complementary_slackness, ord=float("inf")), primal_value,
                                                                               mu, sigma)
    end_time = datetime.datetime.now()
    run_time = end_time - start_time
    print "Solver ran for", run_time
    print "average run_time for each schur complement solve call was", sum(solver_times, datetime.timedelta()) / len(solver_times)
    print "with a total time", sum(solver_times, datetime.timedelta()) 
    print "average run_time for each construction of augmented lagrangian was", sum(augmented_mat_creation_times, datetime.timedelta()) / len(augmented_mat_creation_times)
    print "with a total time", sum(augmented_mat_creation_times, datetime.timedelta())
    if is_solved:
        print "solution found, returning parameters"
        print "primal value is", primal_value
        return x
    else:
        print "failed to find solution, returning None"
        return None
    
def _calculate_mehrotra_step_sizes(slacks, slacks_step, dual_inequality, dual_inequality_step):
    gamma_f = 0.9
    gamma_a = 1. / (1. - gamma_f)
    num_inequality_constraints = slacks.size
    
    max_dual_inequality_step_size, arg_max_dual_inequality_step_size = _max_val_pos_step_size(dual_inequality_step, dual_inequality, return_argmax=True)#, scaling_parameter=tau[iteration])
    max_dual_inequality_step = dual_inequality_step[arg_max_dual_inequality_step_size]
    max_slacks_step_size, arg_max_slacks_step_size = _max_val_pos_step_size(slacks_step, slacks, return_argmax=True)#, scaling_parameter=tau[iteration])
    max_slacks_step = slacks_step[arg_max_slacks_step_size]
    
    mu_full = np.dot(slacks + max_slacks_step_size * slacks_step, 
                     dual_inequality + max_dual_inequality_step_size * dual_inequality_step) / num_inequality_constraints
    mu_full /= gamma_a
    
    max_slacks = slacks[arg_max_slacks_step_size] 
    max_updated_slacks = slacks[arg_max_dual_inequality_step_size] + max_slacks_step_size * slacks_step[arg_max_dual_inequality_step_size] 
    max_dual_inequality = dual_inequality[arg_max_dual_inequality_step_size] 
    max_updated_dual_inequality = dual_inequality[arg_max_slacks_step_size] + max_dual_inequality_step_size * dual_inequality_step[arg_max_slacks_step_size]
    
    
    prov_dual_inequality_step_size = (mu_full / max_updated_slacks - max_dual_inequality) / (max_dual_inequality_step_size * max_dual_inequality_step)
    prov_slacks_step_size = (mu_full / max_updated_dual_inequality - max_slacks) / (max_slacks_step_size * max_slacks_step)
    
#    print (prov_slacks_step_size, prov_dual_inequality_step_size)
    slacks_step_size = max(gamma_f, prov_slacks_step_size) * max_slacks_step_size
    dual_inequality_step_size = max(gamma_f, prov_dual_inequality_step_size) * max_dual_inequality_step_size
#    if slacks_step_size < gamma_f * max_slacks_step_size:
#        slacks_step_size = gamma_f * max_slacks_step_size
#        
#    if dual_inequality_step_size < gamma_f * max_dual_inequality_step_size:
#        dual_inequality_step_size = gamma_f * max_dual_inequality_step_size
    
    slacks_step_size *= .99999999
    dual_inequality_step_size *= .99999999
    
    return slacks_step_size, dual_inequality_step_size

def _calculate_residuals(x, second_order_mat, first_order_vec, A_inequality, A_equality, slacks, dual_inequality, dual_equality):
    residual_lagrangian = np.dot(second_order_mat, x) + first_order_vec - np.dot(A_equality.T, dual_equality) - np.dot(A_inequality.T, dual_inequality)
    residual_inequality = np.dot(A_inequality, x) - b_inequality - slacks
    residual_equality = np.dot(A_equality, x) - b_equality
    residual_complementary_slackness = dual_inequality * slacks
    return residual_lagrangian, residual_inequality, residual_equality, residual_complementary_slackness

def _max_val_pos_step_size(step, a_vec, scaling_parameter=1.0, return_argmax=False):
    """finds step size between 0.0 and 1.0 such that
    a_vec + step_size * step >= (1 - scaling_parameter) e"""
    
    if scaling_parameter > 1.0 or scaling_parameter < 0.0:
        raise ValueError("scaling parameter must be between 0.0 and 1.0")
    
    neg_indices = np.where(step < 0.0)
    if neg_indices[0].size > 0:
        step_size = min(-scaling_parameter * a_vec[neg_indices] / step[neg_indices])
    else:
        step_size = 1.0
    if not return_argmax:
        return step_size
    else:
        arg_step_size = np.where(a_vec / step * -scaling_parameter == step_size)
        return step_size, arg_step_size

def _gondzio_projection(a_vec, min_val, max_val):
    b_vec = copy.deepcopy(a_vec)
    max_index = np.where(a_vec > max_val)
    b_vec[max_index] = max_val
    min_index = np.where(a_vec < min_val)
    b_vec[min_index] = min_val
#    print (np.min(b_vec), np.max(b_vec))
    b_vec -= a_vec
    b_vec[np.where(b_vec < -max_val)] = -max_val
    return b_vec
    

def _schur_complement_solve_symmetric_ul_pd(A, B, C, d, A_is_diag=False, A_inv = None, B_Ainv_BT = None, 
                                            schur_comp_chol_factor = None, neg_schur_comp_chol_factor = None):
    """Solves problem:
        [ A B'] z = d
        [ B C ]
        for z. Assumes that A is a symmetric positive definite, matrix"""
    
    if not A_is_diag:
        if A_inv == None:
            A_inv = sl.inv(A)
        if B_Ainv_BT == None and schur_comp_chol_factor == None and neg_schur_comp_chol_factor == None:
            B_Ainv_BT = FB.dgemm(alpha=1.0, a=np.dot(B, A_inv).T, b=B.T, trans_a=True)
    else:
        if B_Ainv_BT == None and schur_comp_chol_factor == None and neg_schur_comp_chol_factor == None:
            B_Ainv_BT = FB.dgemm(alpha=1.0, a=(B/A[:,np.newaxis]).T, b=B.T, trans_a=True)

    A_dim = A.shape[0]
    z_a= d[:A_dim]
    
    if A_is_diag:
        z_b = d[A_dim:] - np.dot(B, z_a / A)
    else:
        z_b = d[A_dim:] - np.dot(B, np.dot(A_inv, z_a)) #sl.solve(A, z_a))
        
#    if A_is_diag:
#        schur_comp_mat = C - np.dot(B, B.T / A[:,np.newaxis])
#    else:
##        schur_comp_mat = C - np.dot(np.dot(B,A_inv), B.T)
#        schur_comp_mat = C - B_Ainv_BT
    if schur_comp_chol_factor != None:
        z_b = sl.cho_solve(schur_comp_chol_factor, z_b)
    elif neg_schur_comp_chol_factor != None:
        z_b = -sl.cho_solve(neg_schur_comp_chol_factor, z_b)
    else:
        schur_comp_mat = C - B_Ainv_BT    
        z_b = sl.solve(schur_comp_mat, z_b)
    
    if A_is_diag:
        z_a /= A
    else:
        z_a = np.dot(A_inv, z_a) #sl.solve(A, z_a, sym_pos=True, overwrite_b=True)
    #z_b finished, compute z_a
    
    if A_is_diag: 
        z_a -= np.dot(B.T, z_b) / A
    else:
        z_a -= np.dot(A_inv, np.dot(B.T, z_b)) #sl.solve(A, np.dot(B.T, z_b))
        
    return np.reshape(np.hstack((z_a, z_b)), d.shape)

def _schur_complement_solve_symmetric_ld_pd(A, B, C, d, C_is_diag=False, C_inv = None):
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

def ooqp_initialize(G, c, A_equality, b_equality, A_inequality, b_inequality):
    data_vec = np.zeros(6)
    data_vec[0] = np.max(np.abs(G))
    data_vec[1] = np.max(np.abs(c))
    data_vec[2] = np.max(np.abs(A_equality))
    data_vec[3] = np.max(np.abs(b_equality))
    data_vec[4] = np.max(np.abs(A_inequality))
    data_vec[5] = np.max(np.abs(b_inequality))
    
    sqrt_data_norm = np.sqrt(sl.norm(data_vec,ord=float("inf")))
    
    x_0 = np.zeros(len(c))
    slacks_0 = sqrt_data_norm * np.ones(len(b_inequality))
    dual_equality_0 = np.zeros(len(b_equality))
    dual_inequality_0 = sqrt_data_norm * np.ones(len(b_inequality))
    
    return x_0, slacks_0, dual_equality_0, dual_inequality_0

if __name__ == '__main__':
    print "testing interior point algorithm"
    qp_obj = QP_object()
    qp_obj.read_qps_file('qps_files/CVXQP2_M.SIF.txt')
    print "converting to standard form"
    G,c,A_inequality, b_inequality, A_equality, b_equality = qp_obj.qp_obj_to_standard_form_with_equality_constraints()
    x_0, slack_0, dual_equality_0, dual_inequality_0 = ooqp_initialize(G, c, A_equality, b_equality, A_inequality, b_inequality)
    print "running qp"
#    x = interior_point_qp_augmented_lagrangian(G, c, A_equality, b_equality, A_inequality, b_inequality, seed=0, tol=1E-5, max_iterations=150,
    x = interior_point_qp_augmented_lagrangian_multi_corrector(G, c, A_equality, b_equality, A_inequality, b_inequality, 
                                                               seed=1, tol=1E-5, max_iterations=150, max_num_correctors=3,
                                                               x_0=x_0, slack_0=slack_0, dual_inequality_0=dual_inequality_0, 
                                                               dual_equality_0=dual_equality_0)
