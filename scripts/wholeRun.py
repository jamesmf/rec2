# -*- coding: utf-8 -*-
"""
Created on Fri May 6 12:52:21 2016

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
num_folds = 1

print(listdir("../data/raw"))
itemIDs, userIDs   = helper.getVocabularies("../data/raw/ratings.dat")
for fold in range(0,num_folds):
    userRatings,userStats = helper.userRatings("../data/raw/r"+str(fold+1)+".train")
    itemStats    = helper.ratingsAvg("../data/raw/r"+str(fold+1)+".train")
    
    for person in userRatings.keys():
        print(helper.getSample(userRatings[person]))
        past, current = helper.getSample(userRatings[person])
        pastItemInds, currentItemInd, aggregates, currentRating = helper.createExample(past,current,userStats,itemStats,itemIDs,person)
        stop=raw_input()
    

