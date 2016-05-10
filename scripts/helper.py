# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 19:51:39 2015

@author: jmf

Creates documents for each user by:

1) obtaining the mean rating for each movie
2) writes a term in the user's document "like_movieID" or "dislike_movieID"
    - if rating > mean, write like
    - if rating < mean, write dislike
    - write term proportional to stdev above/below mean 
"""

import sys
from os import listdir
from os.path import isfile
import numpy as np
import cPickle

"""iterate over unsplit ratings file to get vocabularies for each embedder"""
def getVocabularies(ratingsFile):
    movieIDs    = {}
    userIDs     = {}

    with open(ratingsFile,'rb') as f:
        for rating in f:
            tsp = rating.split("::")
            if len(tsp) > 1:
                movie   = str(int(tsp[1]))
                user    = tsp[0]
                if user not in userIDs:
                    userIDs[user] = len(userIDs)
                if movie not in movieIDs:
                    movieIDs[movie] = len(movieIDs)
    return movieIDs, userIDs

"""takes in a line from ratings and returns what to write in that user's doc"""    
def writeDoc(fields,ratings,STD_SCALAR):
    movie           = str(int(fields[1]))
    user_rating     = float(fields[2])
    value           = ratings[movie]
    mean_rating     = float(value[0])
    std_rating      = float(value[1])+.00000001
    z               = (user_rating-mean_rating)/std_rating
    writeNum        = z*STD_SCALAR

    #This returns the word to write and how many times to write it in a tuple
    if z > 0:
        out = ("L_"+movie+" ",int(np.floor(writeNum)))
    elif z < 0:
        out = ("D_"+movie+" ",int(np.floor(-writeNum)))
    else:
        out = ("",0)
    return out

"""
gets the mean and stdev rating for each movie
returns a dict of "movie": "[mean,stdev]" 
"""
def ratingsAvg(trainFile):
    movies  = {}
    #dsp = data.split("\n")
    with open(trainFile,'rb') as f:
        for rating in f:
            tsp = rating.split("::")
            if len(tsp) > 1:
                if movies.has_key(tsp[1]):
                    movies[tsp[1]].append(float(tsp[2]))
                else:
                    movies[tsp[1]]  = [float(tsp[2])]
        out ={}            
        for k,v in movies.iteritems():
            out[k]  = str(np.mean(v))+"\t"+str(np.std(v))
        
    return out

"""gets the number of ratings given and the average rating for each user"""
def userRatings(trainFile):
    with open(trainFile,'rb') as f1:
        
        userStats   = {}
        for rating in f1:
            tsp = rating.split("::")
            if len(tsp) > 1:
                if userStats.has_key(tsp[0]):
                    userStats[tsp[0]].append(float(tsp[2]))
                else:
                    userStats[tsp[0]]  = [float(tsp[2])]
        out = {}        
        
        #calculate the number of ratings each user has in training set, then scale
        ratTots     = []
        for k,v in userStats.iteritems():
            ratTots.append(len(v))
        numMean     = np.mean(np.array(ratTots))
        numSTD      = np.std(np.array(ratTots))

        #return dict by userID with mean rating and z-scores on number of ratings given
        for k,v in userStats.iteritems():
            norm    = (len(v) - numMean)/numSTD
            out[k]  = str(np.mean(v))+"\t"+str(norm)
    return out
    
