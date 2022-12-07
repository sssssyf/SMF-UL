import time

import torch
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter
from utils.torch_utils import bias_parameters, weight_parameters, \
    load_checkpoint, save_checkpoint, AdamW
import datetime

class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, train_loader, valid_loader, model, loss_func,
                 _log, save_root, config):
        self._log = _log

        self.cfg = config
        self.save_root = save_root
        self.summary_writer = SummaryWriter(str(save_root))

        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.device, self.device_ids = self._prepare_device(config['n_gpu'],config['device_id'])

        self.model = self._init_model(model)
        self.optimizer = self._create_optimizer()
        self.loss_func = loss_func

        self.best_error = np.inf
        self.i_epoch = 0
        self.i_iter = 0

    @abstractmethod
    def _run_one_epoch(self):
        ...

    @abstractmethod
    def _validate_with_gt(self):
        ...

    def classification_PU(self):
        ...

    def classification_SA(self):
        ...

    def classification_IP(self):
        ...

    def classification_HO13(self):
        ...

    def train(self):
        day = datetime.datetime.now()
        day_str = day.strftime('%m_%d_%H_%M')
        a = time.time()
        b=0
        #f = open( './record/'+str(day_str) + '_' + self.cfg.dataname + '_' + str(self.cfg.epoch_num) +'epoch_interval='+str(self.cfg.val_epoch_size)+ '.txt', 'w')  #记录OA画图
        Best_OA=0
        for epoch in range(self.cfg.epoch_num):
            self._run_one_epoch()
            c=time.time()
            #if self.i_epoch % self.cfg.val_epoch_size == 0:
            #    errors, error_names = self._validate_with_gt()
            #     valid_res = ' '.join(
            #        '{}: {:.2f}'.format(*t) for t in zip(error_names, errors))
            #    self._log.info(' * Epoch {} '.format(self.i_epoch) + valid_res)
            b =b+ time.time()-c

            '''
            if epoch % self.cfg.val_epoch_size == 0:
                if self.cfg.dataname == 'PU':
                    Current_OA=self.classification_PU()
                elif self.cfg.dataname == 'SA':
                    Current_OA=self.classification_SA()
                elif self.cfg.dataname == 'IP':
                    Current_OA=self.classification_IP()
                elif self.cfg.dataname == 'HO13':
                    Current_OA=self.classification_HO13()

                f.write(str(epoch+TargetData)+":"+str(Current_OA) + '\n')
                if Current_OA > Best_OA:
                    Best_OA=Current_OA
                    self.save_model(self.cfg.dataname + '_' + str(self.cfg.epoch_num) +'epoch_interval='+str(self.cfg.val_epoch_size))
            '''
        #f.close()
        #print(b)
        print(time.time()-a)



    def _init_model(self, model):
        model = model.to(self.device)
        if self.cfg.pretrained_model:
            self._log.info("=> using pre-trained weights {}.".format(
                self.cfg.pretrained_model))
            epoch, weights = load_checkpoint(self.cfg.pretrained_model)

            from collections import OrderedDict
            new_weights = OrderedDict()
            model_keys = list(model.state_dict().keys())
            weight_keys = list(weights.keys())
            for a, b in zip(model_keys, weight_keys):
                new_weights[a] = weights[b]
            weights = new_weights
            model.load_state_dict(weights)
        else:
            self._log.info("=> Train from scratch.")
            model.init_weights()
        model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        return model

    def _create_optimizer(self):
        self._log.info('=> setting Adam solver')
        param_groups = [
            {'params': bias_parameters(self.model.module),
             'weight_decay': self.cfg.bias_decay},
            {'params': weight_parameters(self.model.module),
             'weight_decay': self.cfg.weight_decay}]

        if self.cfg.optim == 'adamw':
            optimizer = AdamW(param_groups, self.cfg.lr,
                              betas=(self.cfg.momentum, self.cfg.beta))
        elif self.cfg.optim == 'adam':
            optimizer = torch.optim.Adam(param_groups, self.cfg.lr,
                                         betas=(self.cfg.momentum, self.cfg.beta),
                                         eps=1e-7)
        else:
            raise NotImplementedError(self.cfg.optim)
        return optimizer

    def _prepare_device(self, n_gpu_use,device_id):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self._log.warning("Warning: There\'s no GPU available on this machine,"
                              "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self._log.warning(
                "Warning: The number of GPU\'s configured to use is {}, "
                "but only {} are available.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:'+str(device_id) if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        list_ids = list(range(device_id,device_id+1))

        return device, list_ids


    def save_model(self,  name):
        #is_best = error < self.best_error

        #if is_best:
        #    self.best_error = error



        models = {'epoch': self.i_epoch,
                  'state_dict': self.model.module.state_dict()}


        #save_checkpoint(self.save_root, models, name, is_best)
        save_checkpoint(self.save_root, models, name, is_best=None)
