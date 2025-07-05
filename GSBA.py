import torch.nn as nn
import torchvision.datasets as dsets

import torchvision.transforms as transforms
import torchvision.models as torch_models
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from utils import valid_bounds, clip_image_values
from PIL import Image
from torch.autograd import Variable
from numpy import linalg 
from scipy.fftpack import dct, idct
import math
import copy
import time




class GSBA():
    def __init__(self, model, src_img, mean, std, lb, ub, 
                 tar_img = None, tar_lbls=[None], dim_reduc_factor=4,
                 iteration=150, query_budget=40010, top_k = 1, 
                 base_query=30, tol=0.0001, sigma=0.0002,
                 device='cpu', StepSize=6,
                 query_type = 'nonuniform', b_grad_approach_name = 'Proposed',
                 init_method = 'grad_Proposed'):
        self.model = model
        self.src_img = src_img
        self.src_lbl = torch.argmax(self.model.forward(Variable(self.src_img, requires_grad=True)).data).item()
        if tar_img != None or tar_lbls[0] != None:
            self.tar_img = tar_img
            if tar_img != None:
                pred_score = self.model(tar_img).data
                _, ind = torch.sort(pred_score, descending=True)
                ind = ind.reshape(-1).cpu().numpy()
                self.tar_lbls = ind[0:top_k]
            else:
                self.tar_lbls = tar_lbls
            assert len(self.tar_lbls) == top_k
            print('target labels:', self.tar_lbls)
        else:
            self.tar_lbls = [None]
        self.dim_reduc_factor = dim_reduc_factor
        self.iteration =iteration
        self.top_k = top_k
        self.I0 = base_query
        self.mean = mean
        self.std = std
        self.lb = lb
        self.ub = ub
        self.tol = tol
        self.sigma = sigma
        self.grad_estimator_batch_size = 40
        self.StepSize = StepSize
        self.query_budget = query_budget
        self.device = device
        self.all_queries = 0
        self.query_type = query_type

        self.approach_name = b_grad_approach_name
        self.init_method = init_method
        

        if self.tar_lbls[0] == None:
            self.attack_type = 'untargeted'
        else:
            self.attack_type = 'targeted'
            print(f'Initial boundary finding method: {self.init_method};    \nGrad estimation on boundary approach: {self.approach_name}')

        
        
        
        
    def query_to_classifier(self, x):
        pred_score = self.model(x).data
        val, ind = torch.sort(pred_score, descending=True)
        # self.all_queries += 1
        return val.reshape(-1).cpu().numpy(), ind.reshape(-1).cpu().numpy()



    def is_adversarial(self, x):
        '''Query success indicator'''
        _, ind = self.query_to_classifier(x)
        if self.attack_type == 'untargeted':
            is_adv = self.src_lbl not in ind[:self.top_k]
        else:
            is_adv = set(ind[:self.top_k]) <= set(self.tar_lbls)    # to check if the target label set is a subset of set ind[:self.top_k]
        return is_adv
    

    
    def inv_tf(self, x, mean, std):   
        for i in range(len(mean)):    
            x[i] = np.multiply(x[i], std[i], dtype=np.float32)
            x[i] = np.add(x[i], mean[i], dtype=np.float32)   
        x = np.swapaxes(x, 0, 2)      
        x = np.swapaxes(x, 0, 1)    
        return x
    


    def get_sorted_scores_nd_indices(self, image):
        pred_score = self.model.forward(Variable(image, requires_grad=True)).data
        val, ind = torch.sort(pred_score, descending=True)
        val = F.softmax(val, dim=-1).data
        return val.reshape(-1).cpu().numpy(), ind.reshape(-1).cpu().numpy()
    
    
    def get_scores_nd_indices(self, image):
        predict_label = torch.argmax(self.model.forward(Variable(image, requires_grad=True)).data).item()
        pred_score = self.model.forward(Variable(image, requires_grad=True)).data
        val, ind = torch.sort(pred_score, descending=True)
        return val, ind.reshape(-1)
    
    
    
    def get_socres(self, image):
        pred_score = self.model.forward(Variable(image, requires_grad=True)).data
        val = F.softmax(pred_score, dim=-1).data
        return val.cpu()
    
    
    
    def find_random_adversarial(self, image):
        num_calls = 1       
        step = 0.03
        perturbed = image        
        while self.is_adversarial(perturbed) == 0:           
            pert = torch.randn(image.shape)
            pert = pert.to(self.device)   
            perturbed = image + num_calls*step* pert
            perturbed = clip_image_values(perturbed, self.lb, self.ub)
            perturbed = perturbed.to(self.device)
            num_calls += 1            
        return perturbed, num_calls 
    
    
    
    def bin_search(self, x_0, x_random):  
        num_calls = 0
        adv = x_random
        cln = x_0      
        while True:         
            mid = (cln + adv) / 2.0
            num_calls += 1           
            if self.is_adversarial(mid):
                adv = mid
            else:
                cln = mid   
            if torch.norm(adv-cln).cpu().numpy()<self.tol or num_calls>=100:
                break       
        return adv, num_calls 
    
       



    def grad_estimation_untargeted(self, x, noises):
        '''To estimate gradient on the decision boudnary for the untargeted attack'''
        Xs = x+self.sigma*noises
        batch_size = self.grad_estimator_batch_size
        batch_num = int(np.ceil(Xs.shape[0]/batch_size))
        logits = []
        for k in range(batch_num):
            batch_Xs = Xs[k*batch_size:(k+1)*batch_size]
            logits_Xs = self.model.forward(Variable(batch_Xs, requires_grad=True)).data.cpu()
            logits.append(logits_Xs)
        logits = torch.cat(logits,0)
        # self.all_queries += Xs.shape[0]
                
        pred_scores = F.softmax(logits, dim=-1)
        # print(torch.argmax(pred_scores, 1))
        Pcs = copy.deepcopy(pred_scores[:,self.src_lbl])
        
        pred_scores[:,self.src_lbl] = -torch.inf
        # print(pred_scores.shape)
        sorted_scores, _ = torch.sort(pred_scores, 1, descending=True)
        Pvk = sorted_scores[:, self.top_k-1]
        
        Fs = Pvk - Pcs
        # print('diffs', diffs)
        Fs = Fs[:,None,None,None].to(self.device)
        weighted_noises = Fs*10000*noises
        grad = sum(weighted_noises)            
        if  torch.sum(Fs ==0)==self.I0:
            print('This one working')
            val, ind = torch.sort(logits, descending=True)
            is_advs = [1 if self.src_lbl not in ind[k][:self.top_k].numpy() else -1 for k in range(ind.shape[0])]
            # self.all_queries += self.q0
            is_advs = torch.tensor(is_advs)
            is_advs = is_advs[:, None, None, None].to(self.device)
            weighted_noises = is_advs*noises
            grad = sum(weighted_noises) 
        return grad/torch.norm(grad)


    
    def grad_estimation_targeted(self, x, noises):
        p_cur = self.get_socres(x)
        p_cur = p_cur.view(-1).numpy()
        # self.all_queries -= 1   # The above line is redundant
        
        target_scores_cur = torch.tensor([p_cur[i] for i in self.tar_lbls])
        
        Xs = x+self.sigma*noises
        batch_size = self.grad_estimator_batch_size
        batch_num = int(np.ceil(Xs.shape[0]/batch_size))
        logits = []
        for k in range(batch_num):
            batch_Xs = Xs[k*batch_size:(k+1)*batch_size]
            logits_Xs = self.model.forward(Variable(batch_Xs, requires_grad=True)).data.cpu()
            logits.append(logits_Xs)
        logits = torch.cat(logits,0)

        pred_scores = F.softmax(logits, dim=-1)
        p_pret_targets = pred_scores[:,list(self.tar_lbls)]
        
        _, ind_sort = torch.sort(pred_scores, descending=True)
        is_advs = [1 if set(ind_sort[k][:self.top_k].numpy()) <= set(self.tar_lbls) else -1 for k in range(ind_sort.shape[0])]
        
        if self.approach_name == 'Approach1':   # decision_based
            is_advs = torch.tensor(is_advs)
            is_advs = is_advs[:, None, None, None].to(self.device)
            weighted_noises = is_advs*noises
            grad = sum(weighted_noises)

        elif self.approach_name == 'Approach2':   # all_sum
            xb_scr_sum = torch.sum((target_scores_cur))
            pert_scr_sum = torch.sum(p_pret_targets, dim=1)
            diff = pert_scr_sum - xb_scr_sum
            diff = diff[:, None, None, None].to(self.device)
            weighted_noises = diff*noises
            grad = sum(weighted_noises)
            
        elif self.approach_name == 'Approach3':   # tar_min_diff
            p_pret_targets_cur_min, _ = target_scores_cur.min(0)
            p_pret_targets_min, _ = p_pret_targets.min(1)
            diffs = p_pret_targets_min- p_pret_targets_cur_min
            diffs = diffs[:, None, None, None].to(self.device)
            weighted_noises = diffs*noises
            grad = sum(weighted_noises)

        elif self.approach_name == 'Approach4':   # CW_soft
            tar_prob_min = p_pret_targets.min(dim=1)[0]
            pred_scores[:, list(self.tar_lbls)] = -torch.inf
            max_val = torch.max(pred_scores, dim=1)[0]
            w = tar_prob_min.reshape(-1) - max_val.reshape(-1)
            w = w[:, None, None, None].to(self.device)
            weighted_noises = w*noises
            grad = sum(weighted_noises)

        elif self.approach_name == 'Approach5':   #  CW_logits
            p_pret_logits = logits[:,list(self.tar_lbls)]
            tar_lgt_min = p_pret_logits.min(dim=1)[0]
            logits[:, list(self.tar_lbls)] = -torch.inf
            max_val = torch.max(logits, dim=1)[0]
            w = tar_lgt_min.reshape(-1) - max_val.reshape(-1)
            w = w[:, None, None, None].to(self.device)
            weighted_noises = w*noises
            grad = sum(weighted_noises)

        elif self.approach_name == 'Approach6':   #  Eff_cls_cnt
            diffs = (p_pret_targets - target_scores_cur)
            w = []
            for k in range(diffs.shape[0]):
                wt = torch.sum(is_advs[k]*diffs[k]>0)
                w.append(is_advs[k]*wt)
            w = torch.tensor(w)
            w = w[:,None,None,None].to(self.device)
            weighted_noises = w*noises
            grad = sum(weighted_noises) 

        elif self.approach_name == 'Approach7':   #  Eff_score_sum
            diffs = (p_pret_targets - target_scores_cur)
            w = []
            for k in range(diffs.shape[0]):
                wt = torch.sum(diffs[k]*(is_advs[k]*diffs[k]>0))
                w.append(wt)
            w = torch.tensor(w)
            w = w[:,None,None,None].to(self.device)
            weighted_noises = w*noises
            grad = sum(weighted_noises)

        elif self.approach_name == 'Proposed':
            diffs = (p_pret_targets - target_scores_cur)
            w = []
            for k in range(diffs.shape[0]):
                a = torch.sum(is_advs[k]*diffs[k]>0)
                wt = torch.sum(diffs[k]*(is_advs[k]*diffs[k]>0))
                w.append(a*wt)
            w = torch.tensor(w)
            w = w[:,None,None,None].to(self.device)
            weighted_noises = w*noises
            grad = sum(weighted_noises)

        else:
            NotImplementedError('Approach not implemented')

        return grad/torch.norm(grad)
    
    
    

    def find_next_boundary_targeted(self, x_s, g, x_b):   
        num_calls = 1
        g_hat = g/torch.norm(g)
        psi_hat = (x_b - x_s)/torch.norm(x_b - x_s)
        phi = torch.acos(torch.dot(psi_hat.reshape(-1), g_hat.reshape(-1)))  
        while True:
            alpha = phi/(pow(2, num_calls))
            gamma = (torch.sin(phi)*torch.cos(alpha)/torch.sin(alpha)-torch.cos(phi)).item()
            theta = (g_hat + gamma*psi_hat)/torch.norm(g_hat + gamma*psi_hat)
            perturbed = x_s + theta*torch.norm(x_b-x_s)*torch.dot(theta.reshape(-1), psi_hat.reshape(-1)) 
            if num_calls > 40:
                print('failed ... ')
                return x_b, num_calls
            perturbed = clip_image_values(perturbed, self.lb, self.ub)
            # print(self.is_adversarial(perturbed))
            num_calls += 1
            if self.is_adversarial(perturbed):
                break
        perturbed, bin_query = self.bin_search(self.src_img, perturbed)
        return perturbed, num_calls-1 + bin_query



    def find_next_boundary_untargeted(self, x_s, g, x_b):   
        num_calls = 1
        g_hat = g/torch.norm(g)
        psi_hat = (x_b - x_s)/torch.norm(x_b - x_s)
        phi = torch.acos(torch.dot(psi_hat.reshape(-1), g_hat.reshape(-1))).cpu()       
        while True:  
            alpha = torch.tensor([(math.pi/2)])*(1 - 1/pow(2, num_calls))
            gamma = (torch.sin(phi)*torch.cos(alpha)/torch.sin(alpha)-torch.cos(phi)).item()         
            theta = (g_hat + gamma*psi_hat)/torch.norm(g_hat + gamma*psi_hat)
            xq = x_s + theta*torch.norm(x_b-x_s)*torch.dot(psi_hat.reshape(-1), theta.reshape(-1))
            xq = clip_image_values(xq, self.lb, self.ub)
            if not self.is_adversarial(xq):
                break
            num_calls += 1
            if num_calls > 40:
                print('failed ... ')
                return x_b, num_calls
        perturbed , n_calls = self.SemiCircular_boundary_search(x_s, x_b, xq)
        return perturbed, num_calls+n_calls

    

    def SemiCircular_boundary_search(self, x_0, x_b, p_near_boundary):
        num_calls = 0
        norm_dis = torch.norm(x_b-x_0)
        boundary_dir = (x_b-x_0)/torch.norm(x_b-x_0)
        clean_dir = (p_near_boundary - x_0)/torch.norm(p_near_boundary - x_0)
        adv_dir = boundary_dir
        adv = x_b
        clean = x_0
        while True:
            mid_dir = adv_dir + clean_dir
            mid_dir = mid_dir/torch.norm(mid_dir)
            theta = torch.acos(torch.dot(boundary_dir.reshape(-1), mid_dir.reshape(-1))/ (torch.linalg.norm(boundary_dir)*torch.linalg.norm(mid_dir)))
            d = torch.dot(boundary_dir.reshape(-1), mid_dir.reshape(-1))*norm_dis
            x_mid = x_0 + mid_dir*d
            num_calls +=1
            # print(is_adversarial(x_mid, orig_label))
            if self.is_adversarial(x_mid):
                adv_dir = mid_dir
                adv = x_mid  
            else:
                clean_dir = mid_dir  
                clean = x_mid                             
            if torch.norm(adv-clean).cpu().numpy()<self.tol:
                break
            if num_calls >100:
                break      
        return adv, num_calls
    
    
    
    def find_random(self, x, n):
        image_size = x.shape
        out = torch.zeros(n, 3, int(image_size[-2]), int(image_size[-1]))
        for i in range(n):
            x = torch.zeros(image_size[0], 3, int(image_size[-2]), int(image_size[-1]))
            fill_size = int(image_size[-1]/self.dim_reduc_factor)
            x[:, :, :fill_size, :fill_size] = torch.randn(image_size[0], x.size(1), fill_size, fill_size)
            if self.dim_reduc_factor > 1.0:
                x = torch.from_numpy(idct(idct(x.cpu().numpy(), axis=3, norm='ortho'), axis=2, norm='ortho'))
            out[i] = x
        return out




    def grad_estimation_init_untargeted(self, x, noises):
        '''To estimate gradient in the non-adversarial region for untargeted attacks'''
        p_cur = self.get_socres(x)
        scr_pred_cur = copy.deepcopy(p_cur[:,self.src_lbl])
        p_cur[:,self.src_lbl] = -torch.inf
        sorted_scores_cur, _ = torch.sort(p_cur, 1, descending=True)
        val_cur = sorted_scores_cur[:, self.top_k-1]
        F_cur = val_cur - scr_pred_cur
        
        Xs = x+self.sigma*noises
        batch_size = self.grad_estimator_batch_size
        batch_num = int(np.ceil(Xs.shape[0]/batch_size))
        logits = []
        for k in range(batch_num):
            batch_Xs = Xs[k*batch_size:(k+1)*batch_size]
            logits_Xs = self.model(batch_Xs).data.cpu()
            logits.append(logits_Xs)
        logits = torch.cat(logits,0)
        # self.all_queries += Xs.shape[0]
        
        pred_scores = F.softmax(logits, dim=-1)
        scr_pred = copy.deepcopy(pred_scores[:,self.src_lbl])
        
        pred_scores[:,self.src_lbl] = -torch.inf
        sorted_scores, _ = torch.sort(pred_scores, 1, descending=True)
        val = sorted_scores[:, self.top_k-1]

        F_n = val - scr_pred
        diffs = F_n - F_cur
        diffs = diffs[:,None,None,None].to(self.device)
        weighted_noises = diffs*noises
        grad = sum(weighted_noises)            
        if  torch.sum(diffs ==0)==self.I0:
            val, ind = torch.sort(logits, descending=True)
            is_advs = [1 if self.src_lbl not in ind[k][:self.top_k].numpy() else -1 for k in range(ind.shape[0])]
            # self.all_queries += self.q0
            is_advs = torch.tensor(is_advs)
            is_advs = is_advs[:, None, None, None].to(self.device)
            weighted_noises = is_advs*noises
            grad = sum(weighted_noises)  
        return grad/torch.norm(grad)



    def grad_estimation_Init_targeted(self, x, noises):
        '''To estimate gradient in the non-adversarial region for targeted attacks'''
        p_cur = (self.get_socres(x))
        p_pret_targets_cur = p_cur[:,list(self.tar_lbls)]
        p_pret_targets_cur_min, _ = p_pret_targets_cur.min(1)
        
        Xs = x+self.sigma*noises
        logits = self.model.forward(Variable(Xs, requires_grad=True)).data.cpu()
        
        pred_scores = F.softmax(logits, dim=-1)
        p_pret_targets = pred_scores[:,list(self.tar_lbls)]
        p_pret_targets_min, _ = p_pret_targets.min(1)
        
        if self.init_method == 'grad_Method1':   #  Two_min_diff
            w = p_pret_targets_min- p_pret_targets_cur_min
            w = w[:,None,None,None].to(self.device)
            grad = sum(w*noises)   

        elif self.init_method == 'grad_Method2':   # cw_loss_diff
            p_cur[:,list(self.tar_lbls)] = -torch.inf
            max_val_cur = torch.max(p_cur, dim=1)[0]
            diff_cur = p_pret_targets_cur_min - max_val_cur
            pred_scores[:, list(self.tar_lbls)] = -torch.inf
            max_val = torch.max(pred_scores, dim=1)[0]
            diffs_noise = p_pret_targets_min.reshape(-1) - max_val.reshape(-1)
            w = diffs_noise - diff_cur
            w = torch.tensor(w).to(self.device)
            w = w[:, None, None, None]
            grad = sum(w*noises)

        elif self.init_method == 'grad_Method3':   # eff_cnt_impact
            diff_min = p_pret_targets_min- p_pret_targets_cur_min
            is_eff = [1 if diff_min[k]>0 else -1 for k in range(diff_min.shape[0])]
            diffs = p_pret_targets - p_pret_targets_cur
            w = []
            for k in range(diffs.shape[0]):
                a = torch.sum(is_eff[k]*diffs[k]>0)
                w.append(a*is_eff[k])
            w = torch.tensor(w).to(self.device)
            w = w[:, None, None, None]
            grad = sum(w*noises)

        elif self.init_method == 'grad_Method4':   #   Eff_score_sum
            diff_min = p_pret_targets_min- p_pret_targets_cur_min
            is_eff = [1 if diff_min[k]>0 else -1 for k in range(diff_min.shape[0])]
            diffs = p_pret_targets - p_pret_targets_cur
            w = []
            for k in range(diffs.shape[0]):
                wt = torch.sum(diffs[k]*(is_eff[k]*diffs[k]>0))
                w.append(wt)
            w = torch.tensor(w).to(self.device)
            w = w[:, None, None, None]
            grad = sum(w*noises)

        elif self.init_method == 'grad_Proposed':   #  eff_cnt_wt_impact
            diff_min = p_pret_targets_min- p_pret_targets_cur_min
            is_eff = [1 if diff_min[k]>0 else -1 for k in range(diff_min.shape[0])]
            diffs = p_pret_targets - p_pret_targets_cur
            w = []
            for k in range(diffs.shape[0]):
                a = torch.sum(is_eff[k]*diffs[k]>0)
                wt = torch.sum(diffs[k]*(is_eff[k]*diffs[k]>0))
                w.append(a*wt)
            w = torch.tensor(w).to(self.device)
            w = w[:, None, None, None]
            grad = sum(w*noises)

        else:
            NotImplementedError("Not implemented")

        return grad/torch.norm(grad)
    



    def initial_boundary_with_grad(self, x):
        StepSize=self.StepSize
        print('StepSize:', StepSize)
        num_calls = 1
        while(self.is_adversarial(x)==0):
            noises = self.find_random(x, self.I0)
            if self.attack_type == 'untargeted':
                grad = self.grad_estimation_init_untargeted(x, noises.to(self.device))
            else:
                grad = self.grad_estimation_Init_targeted(x, noises.to(self.device))    
            num_calls += self.I0
            x = x + StepSize*grad
            x = clip_image_values(x, self.lb, self.ub)
            if num_calls >= self.query_budget:
                return x, num_calls, 'failed'
            num_calls += 1
        return x, num_calls, 'success'



    def Attack(self):
        norms = []
        n_query = []
        x_inv = self.inv_tf(self.src_img.cpu()[0,:,:,:].squeeze(), self.mean, self.std)       # to normalize the image from 0 to 1
        
        if 'random' in self.init_method:
            if self.attack_type == 'untargeted':
                x_in_adv, Q_to_boundary= self.find_random_adversarial(self.src_img)
            else:
                x_in_adv, Q_to_boundary= self.tar_img, 0
        else:
            x_in_adv, Q_to_boundary, note = self.initial_boundary_with_grad(self.src_img)
            if note=='failed':
                print(f'Failed to find the initial boundary point with {self.query_budget} query budget')
                return None, [self.query_budget+100], [1000]
        
        x_b, Q_bin = self.bin_search(self.src_img, x_in_adv)                  # initail boundary point
        x_b_inv = self.inv_tf(x_b.cpu()[0,:,:,:].squeeze(), self.mean, self.std) 
        norm_initial = torch.norm(x_b_inv - x_inv)
        
        Q_init = Q_to_boundary + Q_bin
        q_num = Q_init
        print('Initial boundary norm', norm_initial.item())
        print('initial query', Q_init)
        
        _, ind_b = self.get_sorted_scores_nd_indices(x_b)
        print(f'top 10 predictions of the initail boundary are: {ind_b[0:10]}')
        
        norms.append(norm_initial.item())
        n_query.append(q_num)
        size = self.src_img.shape
        
        for i in range(self.iteration):
            if self.query_type == 'uniform':
                I_n = int(self.I0) 
            else:
                I_n = int(self.I0*np.sqrt(i+1))
            if self.dim_reduc_factor < 1.0:
                raise Exception("The dimension reduction factor should be greater than 1 for reduced dimension, and should be 1 for Full dimensional image space.")
            elif self.dim_reduc_factor > 1.0:
                noises = self.find_random(self.src_img, I_n) 
            else:
                noises = torch.randn(I_n,3,size[-2],size[-1]) 
            
            if self.attack_type == 'untargeted':
                grad_oi = self.grad_estimation_untargeted(x_b, noises.to(self.device))
            else:
                grad_oi = self.grad_estimation_targeted(x_b, noises.to(self.device))
            
            q_num = q_num + I_n
            
            if self.attack_type == 'untargeted':
                x_adv, qs = self.find_next_boundary_untargeted(self.src_img, grad_oi, x_b)
            else:
                x_adv, qs = self.find_next_boundary_targeted(self.src_img, grad_oi, x_b)
            q_num = q_num + qs
            
            x_b = x_adv
            x_adv_inv = self.inv_tf(x_adv.cpu()[0,:,:,:].squeeze(), self.mean, self.std)            
            norm = torch.norm(x_inv - x_adv_inv)
            if i%10==0 or i==self.iteration-1:
                print('iteration -> ' + str(i) + '   Queries ' + str(q_num) + ' norm is -> ' + f'{norm.item():.3f}')
                _, ind = self.get_sorted_scores_nd_indices(x_adv)
                print(f'top 10 predictions are: {ind[0:10]}')
                # print(self.all_queries)
            norms.append(norm.item())
            n_query.append(q_num)
            
            if q_num>=self.query_budget:
                print('iteration -> ' + str(i) + '   Queries ' + str(q_num) + ' norm is -> ' + f'{norm.item():.3f}')
                _, ind = self.get_sorted_scores_nd_indices(x_adv)
                print(f'top 10 predictions are: {ind[0:10]}')
                return x_adv, n_query, norms
            
        x_adv = clip_image_values(x_adv, self.lb, self.ub)           
        return x_adv, n_query, norms



    
  
    
  
    
  
    
  
    
  