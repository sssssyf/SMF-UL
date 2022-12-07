from .base_trainer import BaseTrainer
from utils.misc_utils import AverageMeter
from easydict import EasyDict
from torchvision import transforms
from transforms import sep_transforms
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import time




def get_accuracy(y_true, y_pred):
    num_perclass = np.zeros((y_true.max() + 1))
    num = np.zeros((y_true.max() + 1))
    for i in range(len(y_true)):
        num_perclass[y_true[i]] += 1
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            num[y_pred[i]] += 1
    for i in range(len(num)):
        num[i] = num[i] / num_perclass[i]
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    ac = np.zeros((y_true.max() + 1 + 2))
    ac[:y_true.max() + 1] = num
    ac[-1] = acc
    ac[-2] = kappa
    return ac  # acc,num.mean(),kappa


class TestHelper():
    def __init__(self, cfg,model):
        self.cfg = EasyDict(cfg)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.model = model
        self.input_transform = transforms.Compose([
            sep_transforms.Zoom(*self.cfg.test_shape),
            sep_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])


    def run(self, img1,img2):
        img1=self.input_transform(img1).unsqueeze(0)
        img2=self.input_transform(img2).unsqueeze(0)

        img_pair = torch.cat((img1,img2), 1).to(self.device)
        self.model.eval()

        return self.model(img_pair)

class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func,
                 _log, save_root, config):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, _log, save_root, config)

    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()

        key_meter_names = ['Loss', 'l_ph', 'l_sm', 'flow_mean']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        self.model.train()
        end = time.time()

        if 'stage1' in self.cfg:
            if self.i_epoch == self.cfg.stage1.epoch:
                self.loss_func.cfg.update(self.cfg.stage1.loss)

        for i_step, data in enumerate(self.train_loader):
            if i_step > self.cfg.epoch_size:
                break
            # read data to device
            img1, img2 = data['img1'], data['img2']
            img_pair = torch.cat([img1, img2], 1).to(self.device)

            # measure data loading time
            am_data_time.update(time.time() - end)

            # compute output
            res_dict = self.model(img_pair, with_bk=True)
            flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
            flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                     zip(flows_12, flows_21)]
            loss, l_ph, l_sm, flow_mean = self.loss_func(flows, img_pair)

            # update meters
            key_meters.update([loss.item(), l_ph.item(), l_sm.item(), flow_mean.item()],
                              img_pair.size(0))

            # compute gradient and do optimization step
            self.optimizer.zero_grad()
            # loss.backward()

            scaled_loss = 1024. * loss
            scaled_loss.backward()

            for param in [p for p in self.model.parameters() if p.requires_grad]:
                param.grad.data.mul_(1. / 1024)

            self.optimizer.step()

            # measure elapsed time
            am_batch_time.update(time.time() - end)
            end = time.time()

            if self.i_iter % self.cfg.record_freq == 0:
                for v, name in zip(key_meters.val, key_meter_names):
                    self.summary_writer.add_scalar('Train_' + name, v, self.i_iter)

            if self.i_iter % self.cfg.print_freq == 0:
                istr = '{}:{:04d}/{:04d}'.format(
                    self.i_epoch, i_step, self.cfg.epoch_size) + \
                       ' Time {} Data {}'.format(am_batch_time, am_data_time) + \
                       ' Info {}'.format(key_meters)
                self._log.info(istr)

            self.i_iter += 1
        self.i_epoch += 1




