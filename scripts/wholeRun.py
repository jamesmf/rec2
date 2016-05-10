# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:52:21 2016

Runs through k-folds to evaluate recommender system performance using RMSE on the MovieLens dataset

@author: jmf
"""

import numpy as np
import subprocess
import sys
from os import listdir
import helper

#subprocess.call("../data/raw/split_ratings.sh")

num_folds   = 5

print(listdir("../data/raw"))
movieIDs, userIDs   = helper.getVocabularies("../data/raw/ratings.dat")
for fold in range(0,num_folds):
    userRatings     = helper.userRatings("../data/raw/r"+str(fold+1)+".train")
    movieRatings    = helper.ratingsAvg("../data/raw/r"+str(fold+1)+".train")



