import numpy as np
import torch
from torch.utils import data
import pandas as pd
import pdb
from dataparsers.BasicParser import *
import pickle

class CSV1hnnParser(BasicParser):
    def __init__(self, X_file,params):
        super(CSV1hnnParser, self).__init__()
        self.sep = params["sep"]
        self.header = None
        if "header" in params:
            self.header = params["header"]
        self.skiprows = None
        if "skiprows" in params:
            self.skiprows = params["skiprows"]
        self.X = pd.read_csv(X_file, sep=self.sep, header=self.header, skiprows=self.skiprows)
        X_ones_file, X_sign_file, X_meta_file = params["file_ones"], params["file_sign"], params["file_meta"]
        self.Ones = pd.read_csv(X_ones_file, sep=self.sep, header=self.header, skiprows=self.skiprows)
        self.Signs = pd.read_csv(X_sign_file, sep=self.sep, header=None, skiprows=self.skiprows)
        with open(X_meta_file, "rb") as f:
            self.meta = pickle.load(f)
        self.Meta = pd.read_csv(X_file, sep=self.sep, header=self.header, skiprows=self.skiprows)

        #print("initial lines", len(self.X))
        #self.X = self.X.dropna()
        #print("Drop na lines", len(self.X))
        #self.X.index = np.arange(len(self.X))
        self.regression = params["regression"]

        self.label_header = None
        if "label_header" in params:
            self.label_header = params["label_header"]
            self.labels =  self.X[self.label_header]
            del self.X[self.label_header]
        else:
            self.labels = np.zeros(self.X.shape[0])

        #if "ignore_label" in params:
        #    self.labels = np.zeros(self.X.shape[0])

        if "normalizer_const" in params:
            self.X = self.X / params["normalizer_const"]
        #self.center = False
        #if "centering_info" in params:
        #    self.center = True
        #    f = params["centering_info"]
        #    r = np.load(f)
        #    self.mu = r["mu"]
        #    self.std = r["std"]

        #self.use_only = None
        #if "use_only" in params:
        #    self.use_only = int(params["use_only"])

        self.length = self.X.shape[0]
        self.dimension = self.X.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        label = self.labels[index]
        data_point = np.array(self.X.loc[index].values)
        ones = np.array(self.Ones.loc[index].values)
        signs = np.array(self.Signs.loc[index].values)
        i_data = self.meta['idata'][str(signs)]
        #if self.center:
        #    data_point = (data_point - self.mu) / (self.std + 1e-5)
        #if self.use_only is not None:
        #    data_point  = data_point[:self.use_only]
        return data_point, label, ones, signs, i_data

