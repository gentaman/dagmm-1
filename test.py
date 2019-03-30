import numpy as np 
import pandas as pd
import torch
# from data_loader import *
# from main import *
from tqdm import tqdm
from PIL import Image
import os

root = '/home/genta/dataset/DAGM2007/Class1/'
fnames = os.listdir(root)

np_imgs = []
for fname in fnames:
    im = Image.open(os.path.join(root, fname))
    np_imgs.append(np.asarray(im))
    
np_imgs = np.asarray(np_imgs).astype(np.float32)/255.
print(np_imgs.shape)
np_imgs_mean = np_imgs.mean(axis=0)
np_imgs_var = np_imgs.var(axis=0)
np_imgs = (np_imgs - np_imgs_mean) / np.sqrt(np_imgs_var)
np_imgs = np_imgs.astype(np.float64)

np_imgs = np_imgs.reshape(-1, 512*512)

# %load main.py
import os
import argparse
# from solver import Solver
# from data_loader import get_loader
from torch.backends import cudnn
from utils import *

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    mkdir(config.log_path)
    mkdir(config.model_save_path)

    data_loader = get_loader(config.data_path, batch_size=config.batch_size, mode=config.mode)
    
    # Solver
    solver = Solver(data_loader, vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver

# %load solver.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from model import *
import matplotlib.pyplot as plt
from utils import *
# from data_loader import *
import IPython
from tqdm import tqdm

class Solver(object):
    DEFAULTS = {}   
    def __init__(self, data_loader, config):
        # Data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.data_loader = data_loader

        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        # Define model
        self.dagmm = DaGMM(input_size=self.data_loader.dataset.D, n_gmm=self.gmm_k)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.lr)

        # Print networks
        self.print_network(self.dagmm, 'DaGMM')

        if torch.cuda.is_available():
            self.dagmm.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        self.dagmm.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_dagmm.pth'.format(self.pretrained_model))))

        print("phi", self.dagmm.phi,"mu",self.dagmm.mu, "cov",self.dagmm.cov)

        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def reset_grad(self):
        self.dagmm.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def train(self):
        iters_per_epoch = len(self.data_loader)

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0

        # Start training
        iter_ctr = 0
        start_time = time.time()



        self.ap_global_train = np.array([0,0,0])
        for e in range(start, self.num_epochs):
            for i, (input_data, labels) in enumerate(tqdm(self.data_loader)):
                iter_ctr += 1
                start = time.time()

                input_data = self.to_var(input_data)

                total_loss,sample_energy, recon_error, cov_diag = self.dagmm_step(input_data)
                # Logging
                loss = {}
                loss['total_loss'] = total_loss.data.item()
                loss['sample_energy'] = sample_energy.item()
                loss['recon_error'] = recon_error.item()
                loss['cov_diag'] = cov_diag.item()



                # Print out log info
                if (i+1) % self.log_step == -1:
                    elapsed = time.time() - start_time
                    total_time = ((self.num_epochs*iters_per_epoch)-(e*iters_per_epoch+i)) * elapsed/(e*iters_per_epoch+i+1)
                    epoch_time = (iters_per_epoch-i)* elapsed/(e*iters_per_epoch+i+1)
                    
                    epoch_time = str(datetime.timedelta(seconds=epoch_time))
                    total_time = str(datetime.timedelta(seconds=total_time))
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    lr_tmp = []
                    for param_group in self.optimizer.param_groups:
                        lr_tmp.append(param_group['lr'])
                    tmplr = np.squeeze(np.array(lr_tmp))

                    log = "Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}], lr {}".format(
                        elapsed,epoch_time,total_time, e+1, self.num_epochs, i+1, iters_per_epoch, tmplr)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)

                    IPython.display.clear_output()
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)
                    else:
                        plt_ctr = 1
                        if not hasattr(self,"loss_logs"):
                            self.loss_logs = {}
                            for loss_key in loss:
                                self.loss_logs[loss_key] = [loss[loss_key]]
                                plt.subplot(2,2,plt_ctr)
                                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                                plt.legend()
                                plt_ctr += 1
                        else:
                            for loss_key in loss:
                                self.loss_logs[loss_key].append(loss[loss_key])
                                plt.subplot(2,2,plt_ctr)
                                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                                plt.legend()
                                plt_ctr += 1

                        plt.show()

                    print("phi", self.dagmm.phi,"mu",self.dagmm.mu, "cov",self.dagmm.cov)
                # Save model checkpoints
                if (i+1) % self.model_save_step == 0:
                    torch.save(self.dagmm.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_dagmm.pth'.format(e+1, i+1)))

    def dagmm_step(self, input_data):
        self.dagmm.train()
        enc, dec, z, gamma = self.dagmm(input_data)

        total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_function(input_data, dec, z, gamma, self.lambda_energy, self.lambda_cov_diag)

        self.reset_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.dagmm.parameters(), 5)
        self.optimizer.step()

        return total_loss,sample_energy, recon_error, cov_diag

    def test(self):
        print("======================TEST MODE======================")
        self.dagmm.eval()
        self.data_loader.dataset.mode="train"

        N = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0

        for it, (input_data, labels) in enumerate(self.data_loader):
            input_data = self.to_var(input_data)
            enc, dec, z, gamma = self.dagmm(input_data)
            phi, mu, cov = self.dagmm.compute_gmm_params(z, gamma)
            
            batch_gamma_sum = torch.sum(gamma, dim=0)
            
            gamma_sum += batch_gamma_sum
            mu_sum += mu * batch_gamma_sum.unsqueeze(-1) # keep sums of the numerator only
            cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1) # keep sums of the numerator only
            
            N += input_data.size(0)
            
        train_phi = gamma_sum / N
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        print("N:",N)
        print("phi :\n",train_phi)
        print("mu :\n",train_mu)
        print("cov :\n",train_cov)

        train_energy = []
        train_labels = []
        train_z = []
        for it, (input_data, labels) in enumerate(self.data_loader):
            input_data = self.to_var(input_data)
            enc, dec, z, gamma = self.dagmm(input_data)
            sample_energy, cov_diag = self.dagmm.compute_energy(z, phi=train_phi, mu=train_mu, cov=train_cov, size_average=False)
            
            train_energy.append(sample_energy.data.cpu().numpy())
            train_z.append(z.data.cpu().numpy())
            train_labels.append(labels.numpy())


        train_energy = np.concatenate(train_energy,axis=0)
        train_z = np.concatenate(train_z,axis=0)
        train_labels = np.concatenate(train_labels,axis=0)


        self.data_loader.dataset.mode="test"
        test_energy = []
        test_labels = []
        test_z = []
        for it, (input_data, labels) in enumerate(self.data_loader):
            input_data = self.to_var(input_data)
            enc, dec, z, gamma = self.dagmm(input_data)
            sample_energy, cov_diag = self.dagmm.compute_energy(z, size_average=False)
            test_energy.append(sample_energy.data.cpu().numpy())
            test_z.append(z.data.cpu().numpy())
            test_labels.append(labels.numpy())


        test_energy = np.concatenate(test_energy,axis=0)
        test_z = np.concatenate(test_z,axis=0)
        test_labels = np.concatenate(test_labels,axis=0)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        combined_labels = np.concatenate([train_labels, test_labels], axis=0)

        thresh = np.percentile(combined_energy, 100 - 20)
        print("Threshold :", thresh)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score

        accuracy = accuracy_score(gt,pred)
        precision, recall, f_score, support = prf(gt, pred, average='binary')

        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy, precision, recall, f_score))
        
        return accuracy, precision, recall, f_score

# %load data_loader.py
import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
import numpy as np
import collections
import numbers
import math
import pandas as pd
class KDD99Loader(object):
    def __init__(self, data_path, mode="train"):
        self.mode=mode
        if data_path is None:
            data_path = './kdd_cup.npz'
        data = np.load(data_path)

        labels = data["kdd"][:,-1]
        features = data["kdd"][:,:-1]
        N, D = features.shape
        self.D = D
        normal_data = features[labels==1]
        normal_labels = labels[labels==1]

        N_normal = normal_data.shape[0]

        attack_data = features[labels==0]
        attack_labels = labels[labels==0]

        N_attack = attack_data.shape[0]

        randIdx = np.arange(N_attack)
        np.random.shuffle(randIdx)
        N_train = N_attack // 2

        self.train = attack_data[randIdx[:N_train]]
        self.train_labels = attack_labels[randIdx[:N_train]]

        self.test = attack_data[randIdx[N_train:]]
        self.test_labels = attack_labels[randIdx[N_train:]]

        self.test = np.concatenate((self.test, normal_data),axis=0)
        self.test_labels = np.concatenate((self.test_labels, normal_labels),axis=0)


    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return self.train.shape[0]
        else:
            return self.test.shape[0]


    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
            return np.float32(self.test[index]), np.float32(self.test_labels[index])
        

class DAGM2007Loader(KDD99Loader):
    def __init__(self, data_path, mode="train"):
        self.mode=mode
#         data = np.load(data_path)
#         labels = data["kdd"][:,-1]
#         features = data["kdd"][:,:-1]
        features = np_imgs.reshape(np_imgs.shape[0], -1)
        x_train = features[:len(np_imgs)//2]
        x_test = features[len(np_imgs)//2:len(np_imgs)//2+100]
        N, D = features.shape
        y_train = np.ones(len(x_train))
        y_test = np.ones(len(x_test))
        self.D = D
        np.random.seed(2)
        x_test[:5] = np.random.rand(*x_test[:5].shape) + x_train.mean()
        y_test[:5] = 0
#         perm = np.random.permutation(N)
#         labels = np.concatenate((y_train, y_test), axis=0)
#         print(features.shape)
#         print(labels.shape)

#         normal_data = features[labels==1]
#         normal_labels = labels[labels==1]

#         N_normal = normal_data.shape[0]

#         attack_data = features[labels==0]
#         attack_labels = labels[labels==0]

#         N_attack = attack_data.shape[0]

#         randIdx = np.arange(N_attack)
#         np.random.shuffle(randIdx)
#         N_train = N_attack // 2
        self.train = x_train
        self.train_labels = y_train
        
        self.test = x_test
        self.test_labels = y_test

        print(len(self.train))

#         self.train = attack_data[randIdx[:N_train]]
#         self.train_labels = attack_labels[randIdx[:N_train]]

#         self.test = attack_data[randIdx[N_train:]]
#         self.test_labels = attack_labels[randIdx[N_train:]]

#         self.test = np.concatenate((self.test, normal_data),axis=0)
#         self.test_labels = np.concatenate((self.test_labels, normal_labels),axis=0)
    
    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return self.train.shape[0]
        else:
            return self.test.shape[0]


    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
            return np.float32(self.test[index]), np.float32(self.test_labels[index])
    
def get_loader(data_path, batch_size, mode='train', loader=None):
    """Build and return data loader."""

    if loader is not None:
        dataset = loader(data_path, mode)
    else:
        dataset = DAGM2007Loader(data_path, mode=mode)
#         dataset = KDD99Loader(data_path, mode=mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader

# %load model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools
from utils import *

class Cholesky(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        l = torch.potrf(a, False)
        ctx.save_for_backward(l)
        return l
    @staticmethod
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s

class DaGMM(nn.Module):
    """Residual Block."""
    def __init__(self, input_size=118, n_gmm=2, latent_dim=3):
        super(DaGMM, self).__init__()

        layers = []
        layers += [nn.Linear(input_size,60)]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(60,30)]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(30,10)]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(10,1)]

        self.encoder = nn.Sequential(*layers)


        layers = []
        layers += [nn.Linear(1,10)]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(10,30)]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(30,60)]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(60,input_size)]

        self.decoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(latent_dim,10)]
        layers += [nn.Tanh()]        
        layers += [nn.Dropout(p=0.5)]        
        layers += [nn.Linear(10,n_gmm)]
        layers += [nn.Softmax(dim=1)]


        self.estimation = nn.Sequential(*layers)

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm,latent_dim))
        self.register_buffer("cov", torch.zeros(n_gmm,latent_dim,latent_dim))

    def relative_euclidean_distance(self, a, b):
        return (a-b).norm(2, dim=1) / a.norm(2, dim=1)

    def forward(self, x):

        enc = self.encoder(x)

        dec = self.decoder(enc)

        rec_cosine = F.cosine_similarity(x, dec, dim=1)
        rec_euclidean = self.relative_euclidean_distance(x, dec)

        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)

        gamma = self.estimation(z)

        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        self.phi = phi.data

 
        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K

        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov
        
    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)

        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))
        print(z_mu)

        eps = 1e-6
        cov_K = cov + to_var(torch.cat([torch.eye(D).unsqueeze(0) for i in range(k)])) * eps
        L = torch.cholesky(cov_K)
        v = torch.einsum('ijk,jnk->ijn', z_mu, torch.inverse(L))
        det_cov = torch.prod(L[:, range(D), range(D)], dim=1)
        cov_diag = torch.sum(1 / cov_K[:, range(D), range(D)])

        # N x K
        exp_term_tmp = -0.5 * torch.einsum('ijk,ijk->ij', v, v)
    
        dlog = D * np.log(2*np.pi)
        sample_energy = - torch.logsumexp(torch.log(phi.unsqueeze(0)) + exp_term_tmp - 0.5*(dlog + torch.log(det_cov)), dim=1)


        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag


    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):

        recon_error = torch.mean((x - x_hat) ** 2)

        phi, mu, cov = self.compute_gmm_params(z, gamma)

        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)

        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag

        return loss, sample_energy, recon_error, cov_diag


class hyperparams():
    def __init__(self, config):
        self.__dict__.update(**config)
defaults = {
    'lr' : 1e-4,
    'num_epochs' : 3,
    'batch_size' : 100//4,
    'gmm_k' : 4,
    'lambda_energy' : 0.1,
    'lambda_cov_diag' : 0.005,
    'pretrained_model' : None,
    'mode' : 'train',
    'use_tensorboard' : False,
    'data_path' : 'kdd_cup.npz',

    'log_path' : './dagmm/logs',
    'model_save_path' : './dagmm/models',
    'sample_path' : './dagmm/samples',
    'test_sample_path' : './dagmm/test_samples',
    'result_path' : './dagmm/results',

    'log_step' : 194//4,
    'sample_step' : 194,
    'model_save_step' : 194,
}

def test_loop(defaults, log=None):
    solver = main(hyperparams(defaults))
    e1 = time.time()
    accuracy, precision, recall, f_score = solver.test()
    e2 = time.time()
    if isinstance(log, dict):
        log['train'] = [accuracy,precision, recall, f_score]
        log['time'] = e2 - e1


    solver.data_loader.dataset.mode="train"
    solver.dagmm.eval()
    N = 0
    mu_sum = 0
    cov_sum = 0
    gamma_sum = 0

    for it, (input_data, labels) in enumerate(solver.data_loader):
        input_data = solver.to_var(input_data)
        enc, dec, z, gamma = solver.dagmm(input_data)
        phi, mu, cov = solver.dagmm.compute_gmm_params(z, gamma)

        batch_gamma_sum = torch.sum(gamma, dim=0)

        gamma_sum += batch_gamma_sum
        mu_sum += mu * batch_gamma_sum.unsqueeze(-1) # keep sums of the numerator only
        cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1) # keep sums of the numerator only

        N += input_data.size(0)

    train_phi = gamma_sum / N
    train_mu = mu_sum / gamma_sum.unsqueeze(-1)
    train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

    print("N:",N)
    print("phi :\n",train_phi)
    print("mu :\n",train_mu)
    print("cov :\n",train_cov)
    if isinstance(log, dict):
        log['N'] = N
        log['phi'] = train_phi
        log['mu'] = train_mu
        log['cov'] = train_cov

    train_energy = []
    train_labels = []
    train_z = []
    for it, (input_data, labels) in enumerate(solver.data_loader):
        input_data = solver.to_var(input_data)
        enc, dec, z, gamma = solver.dagmm(input_data)
        sample_energy, cov_diag = solver.dagmm.compute_energy(z, phi=train_phi, mu=train_mu, cov=train_cov, size_average=False)

        train_energy.append(sample_energy.data.cpu().numpy())
        train_z.append(z.data.cpu().numpy())
        train_labels.append(labels.numpy())


    train_energy = np.concatenate(train_energy,axis=0)
    train_z = np.concatenate(train_z,axis=0)
    train_labels = np.concatenate(train_labels,axis=0)

    solver.data_loader.dataset.mode="test"
    test_energy = []
    test_labels = []
    test_z = []
    for it, (input_data, labels) in enumerate(solver.data_loader):
        input_data = solver.to_var(input_data)
        enc, dec, z, gamma = solver.dagmm(input_data)
        sample_energy, cov_diag = solver.dagmm.compute_energy(z, size_average=False)
        test_energy.append(sample_energy.data.cpu().numpy())
        test_z.append(z.data.cpu().numpy())
        test_labels.append(labels.numpy())


    test_energy = np.concatenate(test_energy,axis=0)
    test_z = np.concatenate(test_z,axis=0)
    test_labels = np.concatenate(test_labels,axis=0)

    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    combined_z = np.concatenate([train_z, test_z], axis=0)
    combined_labels = np.concatenate([train_labels, test_labels], axis=0)

    thresh = np.percentile(combined_energy, 100 - 5/600*100)
    # print("Threshold :", thresh)

    # 逆
    pred = (test_energy<thresh).astype(int)
    gt = test_labels.astype(int)

    from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score

    accuracy = accuracy_score(gt,pred)
    precision, recall, f_score, support = prf(gt, pred, average='binary')
    if isinstance(log, dict):
        log['test'] = [accuracy,precision, recall, f_score]

    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy,precision, recall, f_score))

    energy = test_energy
    anomaly_index = test_labels == 0
    normal_index = test_labels == 1
    data = solver.data_loader.dataset.test


    # plt.figure(figsize=[16,6])
    # histinfo = plt.hist(energy, bins=50)
    # plt.xlabel("DAGMM Energy")
    # plt.ylabel("Number of Sample(s)")
    # plt.savefig("./dagm2007_class1_energy_hist.png")
    # plt.show()

    # plt.figure(figsize=[16,6])
    # normal_energy = energy.copy()
    # anomaly_energy = energy.copy()
    # normal_energy[anomaly_index] = normal_energy[normal_index].min()
    # anomaly_energy[normal_index] = normal_energy[normal_index].min()
    # plt.plot(normal_energy, "o-", c='b')
    # plt.plot(anomaly_energy, "x-", c='g')
    # plt.hlines(np.percentile(energy, 95), 0, 100, 'g', linestyles='dashed', label='anomaly ')
    # plt.xlabel("Index (row) of Sample")
    # plt.ylabel("Energy")
    # plt.savefig("./dagm2007_class1_energy.png")
    # plt.show()
    del solver.dagmm
    del solver.data_loader
    del solver
    gc.collect()
    return log


defaults = {
    'lr' : 1e-4,
    'num_epochs' : 3,
    'batch_size' : 100//4,
    'gmm_k' : 4,
    'lambda_energy' : 0.1,
    'lambda_cov_diag' : 0.005,
    'pretrained_model' : None,
    'mode' : 'train',
    'use_tensorboard' : False,
    'data_path' : 'kdd_cup.npz',

    'log_path' : './dagmm/logs',
    'model_save_path' : './dagmm/models',
    'sample_path' : './dagmm/samples',
    'test_sample_path' : './dagmm/test_samples',
    'result_path' : './dagmm/results',

    'log_step' : 194//4,
    'sample_step' : 194,
    'model_save_step' : 194,
}
logs = []
import gc

for batch_size in range(2, 101, 4):
    defaults['batch_size'] = batch_size
    try:
        log = test_loop(defaults, log={})
    except RuntimeError:
        print('runtime error :', batch_size)
        break
    gc.collect()
    logs.append(log)

import pickle

with open('test.log', mode='wb') as f:
    pickle.dump(logs, f)
