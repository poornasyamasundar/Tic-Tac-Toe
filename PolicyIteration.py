import linecache
import os.path
import random
import numpy as np
from helper import *
from ValueIteration import *
import sys

#get the polices after each policy iteration
def getPoliciesPolicyIteration(n, winReward, loseReward, tieReward, agentSide, gamma):
    games = getGames(3)
    l = []
    policies = []
    pi = getPolicyVector(3, agentSide)
    #initially append the arbitrary policy generated
    policies.append(pi)
    validStateIndices = getValidStateIndices(agentSide)
    for i in range(n):
        sys.stdout.write('\r')
        s = "computing Vpi from pi iteration "+ str(i) + "        "
        sys.stdout.write(s)
        sys.stdout.flush()
        print()
        W = getVPi( policies[i], validStateIndices, winReward, loseReward, tieReward, games, agentSide, gamma )  #get VPi from Pi
        sys.stdout.write('\r')
        s = "computing Pi from Vpi iteration "+ str(i) + "        "
        sys.stdout.write(s)
        sys.stdout.flush()
        print()
        pistar = getPiStar(W, getRewardInfo(3, winReward, loseReward, tieReward, games, agentSide), validStateIndices, winReward, tieReward, games, 3, agentSide, gamma) #extract pi from Vpi
        policy = []
        for j in range(19683):
            policy.append(0)
        for j in range(len(pistar)):
            policy[validStateIndices[j]] = pistar[j]
        policies.append(policy)

    return policies

#returns VPi given a Pi
def getVPi( Pi, validStateIndices, winReward, loseReward, tieReward, games, agentSide, gamma ):
    policy = []
    for i in range(len(validStateIndices)):
        policy.append(Pi[validStateIndices[i]])
    Ppi = getPpi( policy, validStateIndices, games, agentSide)
    Ppi = np.array(Ppi)

    gamma_P = np.multiply(Ppi, gamma)
    Identity = np.identity(len(validStateIndices), dtype = float)
    I_minus_gamma_P = np.subtract(Identity, gamma_P)

    Inverse = np.linalg.inv(I_minus_gamma_P)

    Rbar = getRbar(getRewardInfo(3, winReward, loseReward, tieReward, games, agentSide), policy, validStateIndices, winReward, tieReward, games, agentSide)
    Rbar = np.array(Rbar)
    Vpi = np.dot(Inverse, Rbar)
    Vpi = Vpi.tolist()
    return Vpi

#get the Probability 2d array for a pi
def getPpi( Pi, validStateIndices, games, agentSide ):
    Ppi = []
    n = len(Pi)
    for i in range(len(Pi)):
        j = (i+1)/n
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%     " % ('#'*int(20*j), 100*j))
        sys.stdout.flush()
        Ppi.append( getPpiX(i, Pi[i], validStateIndices, games, agentSide))
    return Ppi

#get P[X][.] given X and action
def getPpiX( XIndex, action, validStateIndices, games, agentSide ):
    piX = []
    for i in range(len(validStateIndices)):
        piX.append(0)
    XIndex = validStateIndices[XIndex]
    state = getConfig( XIndex, 3)
    if( agentSide == 'X' ):
        state[action[0]][action[1]] = 1
    else:
        state[action[0]][action[1]] = 2
    newStateIndex = getIndex(3, state)

    game_over = games[getIndex(3, state)]
    if( game_over[0] == True and game_over[2] == False ):
        piX[0] = 1
        return piX
    elif( game_over[0] == True and game_over[2] == True ):
        piX[0] = 1
        return piX

    if( agentSide == 'X' ):
        opponentFilename = 'opponent'+str(3)+'O.txt'
    else:
        opponentFilename = 'opponent'+str(3)+'X.txt'
    listOfProb = get_int_list(linecache.getline(opponentFilename, newStateIndex+1))
    if( listOfProb == [] ):
        print("Error list of Prob is empty")
        print("state index = ", newStateIndex)
        print("state = ")
        print(state[0])
        print(state[1])
        print(state[2])
    newListOfProb = getNormalizedProb(listOfProb)

    for i in range(len(newListOfProb)):
        sampleState = []
        for m in range(3):
            k = []
            for j in range(3):
                k.append(state[m][j])
            sampleState.append(k)
        if( agentSide == 'X' ):
            sampleState[i//3][i%3] = 2
        else:
            sampleState[i//3][i%3] = 1
        n = getIndex(3, sampleState)
        for j  in range(len(validStateIndices)):
            if( validStateIndices[j] == n ):
                piX[j] = newListOfProb[i]
                break

    return piX

#get Rbar from Pi
def getRbar( rewardInfo, Pi, validStateIndices, winReward, tieReward, games, agentSide ):
    Rbar = []
    for i in range(len(Pi)):
        Rbar.append(getRbarX( rewardInfo, i, Pi[i], validStateIndices, winReward, tieReward, games, agentSide))
    return Rbar

#get RbarX 
def getRbarX( rewardInfo, XIndex, action, validStateIndices, winReward, tieReward, games, agentSide ):
    RbarX = 0
    XIndex = validStateIndices[XIndex]
    state = getConfig( XIndex, 3)
    if( agentSide == 'X' ):
        state[action[0]][action[1]] = 1
    else:
        state[action[0]][action[1]] = 2
    newStateIndex = getIndex(3, state)
    game_over = games[getIndex(3,state)]
    
    if( game_over[0] == True and game_over[2] == False ):
        return winReward
    elif( game_over[0] == True and game_over[2] == True ):
        return tieReward

    if( agentSide == 'X' ):
        opponentFilename = 'opponent'+str(3)+'O.txt'
    else:
        opponentFilename = 'opponent'+str(3)+'X.txt'
    listOfProb = get_int_list(linecache.getline(opponentFilename, newStateIndex+1))
    newListOfProb = getNormalizedProb(listOfProb)

    listOfRewards = []
    for i in range(len(newListOfProb)):
        sampleState = []
        for m in range(3):
            k = []
            for j in range(3):
                k.append(state[m][j])
            sampleState.append(k)
        sampleState[i//3][i%3] = 2
        
        n = getIndex(3, sampleState)
        RbarX = RbarX + rewardInfo[n] * newListOfProb[i]

    return RbarX

    











          
    
