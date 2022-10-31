import torch
from torch import nn
from torch.autograd import Variable
from Data import Data
from Model import Model
from Optimizer import Optimizer
from Loss import Loss
from ProgressEvaluator import ProgressEvaluator
import pdb
import numpy as np
from tqdm import tqdm
import pandas as pd
from models.list_models import RoastLinear

class TLoop:
    def __init__(self, params):
        if "merge_logic" in params:
            self.merge_logic = params["merge_logic"]
        else:
            self.merge_logic = None
        self.device_id = params["device_id"]
        self.epochs = params["epochs"]
        # data
        self.train_data = Data(params["train_data"])
        self.progress_test_data = Data(params["progress_test_data"])
        datas = {"valid" : self.progress_test_data}
        self.test_data = None
        if "test_data" in params:
            self.test_data = Data(params["test_data"])
            datas['test'] = self.test_data
        self.progress_train_data = None
        if "progress_train_data" in params:
            self.progress_train_data = Data(params["progress_train_data"])
            datas['train'] = self.progress_train_data
        # model
        mparams = params["model"]
        mparams[mparams["name"]]["seed"] = 101
        mparams[mparams["name"]]["compression"] = 0.8
        self.model1 = Model.get(mparams)
        mparams[mparams["name"]]["seed"] = 105
        mparams[mparams["name"]]["compression"] = 0.6
        self.model2 = Model.get(mparams)
  
        if self.device_id != -1:
          self.model1 = self.model1.cuda(self.device_id)
          self.model2 = self.model2.cuda(self.device_id)
        # optimizer
        self.optimizer1 = Optimizer.get(self.model1, params["optimizer"])
        self.optimizer2 = Optimizer.get(self.model2, params["optimizer"])
        # loss
        self.loss_func = Loss.get(params["loss"])
        #if self.device_id != -1:
        #  self.loss_func = self.loss_func.cuda(self.device_id)
        # progress evaluator
        self.progress_evaluator = ProgressEvaluator.get(params["progress_evaluator"], datas, self.device_id)
        self.metrics = []
        if "metrics" in params:
            self.metrics = params["metrics"].split(",")
        self.binary = False
        if "binary" in params:
            self.binary = params["binary"]

        self.regression = False
        if "regression" in params:
            self.regression = params["regression"]

        self.quiet = False
        if "quiet" in params:
            self.quiet = params["quiet"]

        self.model_internal_logging_itr = -1
        if "model_internal_logging_itr" in params:
            self.model_internal_logging_itr = params["model_internal_logging_itr"]
        self.model_log_file = "./model_log.csv"
        if "model_log_file" in params:
            self.model_log_file = params["model_log_file"]
        self.set_full_data_in_model = False
        if "set_full_data_in_model" in params:
            self.set_full_data_in_model = params["set_full_data_in_model"]
        print(self.model1)
        print(self.model2)

    def merge_logic_1(self, alpha=0.5):
        dic1, dic2 = {}, {}
        for n, m in self.model1.named_modules():
            if type(m) == RoastLinear:
                dic1[n] = (m.grad_comp_to_orig(m.weight.grad), m.bias.grad)
          
        for n, m in self.model2.named_modules():
            if type(m) == RoastLinear:
                dic2[n] = (m.grad_comp_to_orig(m.weight.grad), m.bias.grad)

        assert(sorted(dic1.keys()) == sorted(dic2.keys()))

        for n, m in self.model1.named_modules():
            if type(m) == RoastLinear:
                grad_update_w, grad_update_b = m.grad_orig_to_comp(dic2[n][0]), dic2[n][1]
                m.weight.grad = alpha * m.weight.grad + (1 - alpha) * grad_update_w
                m.bias.grad = alpha * m.bias.grad + ( 1 - alpha) * grad_update_b
          
        for n, m in self.model2.named_modules():
            if type(m) == RoastLinear:
                grad_update_w, grad_update_b = m.grad_orig_to_comp(dic1[n][0]), dic1[n][1]
                m.weight.grad = alpha * m.weight.grad + (1 - alpha) * grad_update_w
                m.bias.grad = alpha * m.bias.grad + (1 - alpha) * grad_update_b


    def merge_logic_2(self, alpha=0.5):
        dic1, dic2 = {}, {}
        for n, m in self.model1.named_modules():
            if type(m) == RoastLinear:
                dic1[n] = (m.wt_comp_to_orig(m.weight.data), m.bias.data)
          
        for n, m in self.model2.named_modules():
            if type(m) == RoastLinear:
                dic2[n] = (m.grad_comp_to_orig(m.weight.data), m.bias.data)

        assert(sorted(dic1.keys()) == sorted(dic2.keys()))

        for n, m in self.model1.named_modules():
            if type(m) == RoastLinear:
                wt_update_w, wt_update_b = m.wt_orig_to_comp(dic2[n][0]), dic2[n][1]
                m.weight.data = alpha * m.weight.data + (1 - alpha) * wt_update_w
                m.bias.data = alpha * m.bias.data + (1 - alpha) * wt_update_b
          
        for n, m in self.model2.named_modules():
            if type(m) == RoastLinear:
                wt_update_w, wt_update_b = m.wt_orig_to_comp(dic1[n][0]), dic1[n][1]
                m.weight.data = alpha * m.weight.data + (1 - alpha) * wt_update_w
                m.bias.data = alpha * m.bias.data + (1 - alpha) * wt_update_b

    def get_complete_data(self):
        self.train_data.reset()
        num_samples = self.train_data.len()
        batch_size = self.train_data.batch_size()
        num_batches = int(np.ceil(num_samples/batch_size))
        xs = []
        ys = []
        for i in tqdm(range(num_batches), disable=self.quiet):
            if self.train_data.end():
              break
            x, y = self.train_data.next()
            xs.append(x)
            ys.append(y)

        return torch.cat(xs, dim=0).cuda(self.device_id), torch.cat(ys, dim=0).cuda(self.device_id)
    

    def loop(self):
        epoch = 0
        iteration = 0
        if self.set_full_data_in_model:
            x_data, y_data = self.get_complete_data()
            self.model.set_data(x_data, y_data)

        while epoch < self.epochs :
            self.train_data.reset()
            num_samples = self.train_data.len()
            batch_size = self.train_data.batch_size()
            num_batches = int(np.ceil(num_samples/batch_size))
            loc_itr = 0
            print("1", self.model1.last_layer.bias)
            print("2", self.model2.last_layer.bias)
            for i in tqdm(range(num_batches), disable=self.quiet):
                if self.train_data.end():
                  break
                self.model1.train()
                self.optimizer1.zero_grad()
                self.model2.train()
                self.optimizer2.zero_grad()

                x, y = self.train_data.next()
                x1 = x[y < 5]
                x2 = x[y >=5]
                y1 = y[y < 5]
                y2 = y[y >=5]

                x1 = Variable(x1).cuda(self.device_id) if self.device_id!=-1 else Variable(x1)
                y1 = Variable(y1).cuda(self.device_id) if self.device_id!=-1 else Variable(y1)

                x2 = Variable(x2).cuda(self.device_id) if self.device_id!=-1 else Variable(x2)
                y2 = Variable(y2).cuda(self.device_id) if self.device_id!=-1 else Variable(y2)
                output1 = self.model1(x1)
                output2 = self.model2(x2)

                if (self.set_full_data_in_model):
                    y = self.model.y_data()
                if self.binary or self.regression:
                    loss1 = self.loss_func(output1.view(-1), y1.float())
                    loss2 = self.loss_func(output2.view(-1), y2.float())
                else:
                    loss1 = self.loss_func(output1, y1)
                    loss2 = self.loss_func(output2, y2)
                loss1.backward()
                loss2.backward()
                
                if self.merge_logic == "merge-grad":
                    self.merge_logic_1()
                    self.optimizer1.step()
                    self.optimizer2.step()
                elif self.merge_logic == "merge-wt":
                    self.optimizer1.step()
                    self.optimizer2.step()
                    self.merge_logic_2()
                else:
                    self.optimizer1.step()
                    self.optimizer2.step()

                #print("-1-", torch.sum(self.model1.first_layer.weight))
                #print("-2-", torch.sum(self.model2.first_layer.weight))

                self.progress_evaluator.evaluate(epoch, loc_itr, iteration, self.model1, self.loss_func, metrics=self.metrics, binary=self.binary, regression=self.regression, split_label=True)
                self.progress_evaluator.evaluate(epoch, loc_itr, iteration, self.model2, self.loss_func, metrics=self.metrics, binary=self.binary, regression=self.regression, split_label=True)

                if self.model_internal_logging_itr > 0 and iteration % self.model_internal_logging_itr == 0:
                    self.model.logger(iteration, True)
                    logdata1 = self.model1.get_logged_data(True)
                    logdata2 = self.model2.get_logged_data(True)
                    if len(logdata1['iterations']) > 0:
                        df = pd.DataFrame(logdata1)
                        df.to_csv(self.model_log_file.strip(".csv") + "1.csv", index=False)
                    if len(logdata2['iterations']) > 0:
                        df = pd.DataFrame(logdata1)
                        df.to_csv(self.model_log_file.strip(".csv") + "2.csv", index=False)
                iteration = iteration + 1
                loc_itr = loc_itr + 1
                #print("Loss", loss)
            epoch = epoch + 1

        self.progress_evaluator.evaluate(epoch, loc_itr, iteration, self.model1, self.loss_func, metrics=self.metrics, binary=self.binary, regression=self.regression, split_label=True)
        self.progress_evaluator.evaluate(epoch, loc_itr, iteration, self.model2, self.loss_func, metrics=self.metrics, binary=self.binary, regression=self.regression, split_label=True)

        if self.model_internal_logging_itr > 0 and iteration % self.model_internal_logging_itr == 0:
            self.model.logger(iteration, True)
            logdata1 = self.model1.get_logged_data(True)
            logdata2 = self.model2.get_logged_data(True)
            if len(logdata1['iterations']) > 0:
                df = pd.DataFrame(logdata1)
                df.to_csv(self.model_log_file.strip(".csv") + "1.csv", index=False)
            if len(logdata2['iterations']) > 0:
                df = pd.DataFrame(logdata1)
                df.to_csv(self.model_log_file.strip(".csv") + "2.csv", index=False)
