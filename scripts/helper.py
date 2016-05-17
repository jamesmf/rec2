# -*- coding: utf-8 -*-
"""
Created on Fri May 6 19:51:39 2016

@author: frickjm

utility functions for RNN recommender systems
"""

import sys
from os import listdir
from os.path import isfile
import numpy as np
import cPickle


def getVocabularies(ratingsFile):
    """
    iterate over unsplit ratings file to get vocabularies for each embedder
    
    Returns:
        itemIds:
            dictionary of {'item_ID':'index'}
            
        userIDs:
            dictionary of {'user_ID':'index'}
    """
    itemIDs    = {}
    userIDs     = {}

    with open(ratingsFile,'rb') as f:
        for rating in f:
            tsp = rating.split("::")
            if len(tsp) > 1:
                item   = int(tsp[1])
                user    = int(tsp[0])
                if user not in userIDs:
                    userIDs[user] = len(userIDs)
                if item not in itemIDs:
                    itemIDs[item] = len(itemIDs)
    return itemIDs, userIDs



def ratingsAvg(trainFile):
    """
    gets the mean and stdev rating for each item
    returns a dict of "item": "[mean,stdev]" 
    """
    items  = {}
    #dsp = data.split("\n")
    with open(trainFile,'rb') as f:
        for rating in f:
            tsp = rating.split("::")
            if len(tsp) > 1:
                if items.has_key(int(tsp[1])):
                    items[int(tsp[1])].append(float(tsp[2]))
                else:
                    items[int(tsp[1])]  = [float(tsp[2])]
        out ={}            
        for k,v in items.iteritems():
            out[k]  = [np.mean(v), np.std(v)]
        
    return out
  

def userRatings(trainFile):
    """
    Returns two dicts indexed by user_id
    
    userRatings:
        contains all a user's ratings in the form [item_ID, rating]
        
    userStats:
        contains a user's aggregate info (based only on training set)
        [user_rating_mean, user_rating_std, user_num_ratings]
        
    Note: 
        more information can easily be added to userStats to increase the "content 
        based" aspect of this approach
    """
    
    with open(trainFile,'rb') as f1:
        userRatings   = {}
        for rating in f1:
            tsp = rating.split("::")
            if len(tsp) > 1:
                if userRatings.has_key(int(tsp[0])):
                    userRatings[int(tsp[0])].append(np.array([int(tsp[1]),float(tsp[2])]))
                else:
                    userRatings[int(tsp[0])]  = [np.array([int(tsp[1]),float(tsp[2])])]
        userStats = {}        
        
        #calculate the number of ratings each user has in training set, then scale
        ratTots     = []
        for k,v in userRatings.iteritems():
            ratTots.append(len(v))
        numMean     = np.mean(np.array(ratTots))
        numSTD      = np.std(np.array(ratTots))

        #return dict by userID with mean rating and z-scores on number of ratings given
        for k,v in userRatings.iteritems():
            norm    = (len(v) - numMean)/numSTD #number of items rated (z-score)
            rats = [r[1] for r in v]
            rmean = np.mean(rats) #mean rating for user
            rstd = np.std(rats) #std of user's ratings
            userStats[k]  = np.array([rmean,rstd, norm])
    return userRatings, userStats
    

def getSample(toSample,numToPick=11):
    if type(toSample) != np.ndarray:
        toSample = np.array(toSample)
    inds = np.random.choice(toSample.shape[0],numToPick,replace=False)
    ts = toSample[inds]
    past = ts[:-1]
    current = ts[-1]
    return past, current
    
def createExample(past,current,userStats,itemStats,itemIDs,userID,predict=False):
    pastItemInds     = [itemIDs[int(i[0])] for i in past]
    pastRatings = [float(i[1]) for i in past]
    print(pastItemInds)
    currentItemInd = itemIDs[int(current[0])]*len(pastItemInds)
    print(currentItemInd)
    if not predict:
        currentRating = itemIDs[int(current[1])]
    else:
        currentRating = None
    
    #building the aggregate information vector - starting with item information
    aggregates = [np.array(itemStats[int(i[0])]) for i in past]

    #add user information to the vector
    userInfo = list(userStats[userID])
    aggregates = [np.append(agg,userInfo,axis=0) for agg in aggregates]    

    #add current item information to the vector
    itemInfo = list(itemStats[int(current[0])])
    aggregates = [np.append(agg,itemInfo,axis=0) for agg in aggregates] 
    
    #add past rating information for each item to aggregate vector
    aggregates = [np.append(agg,pastRatings[i]) for i,agg in enumerate(aggregates)] 
    print(aggregates)
    
    return pastItemInds, currentItemInd, aggregates, currentRating