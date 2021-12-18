from helper import *
from opponent import *
import linecache
import os.path
import random
import sys

"""
    This function returns an array of rewards for each state.
"""
def  getRewardInfo(boardSize, winReward, loseReward, tieReward, games, agentSide ):
    info = []

    #for all board Configurations
    for i in range(3**(boardSize*boardSize)):

        #get the board winners
        s = games[i]

        #if game completes
        if( s[0] == True ):             
            if( s[2] == True ):             
                info.append(tieReward)
            else:
                if( agentSide == 'X' ):
                    if( s[1] == 1 ):
                        info.append(winReward)    
                    else:
                        info.append(loseReward)
                else:
                    if( s[1] == 1 ):
                        info.append(loseReward)
                    else:
                        info.append(winReward)
        else:
            #if the state is not terminal, reward is 0
            info.append(0)
    return info

"""
    This function takes cumulative probabilites and returns the normalized list of probabilites
"""
def getNormalizedProb(listOfProb):
    last = 0
    newListOfProb = []
    newListOfProb.append(listOfProb[0])
    for i in range(8):
        if( listOfProb[i+1] != 0 ):
            newListOfProb.append( listOfProb[i+1] - listOfProb[last])
            last = i+1
        else:
            newListOfProb.append(0)
    sumOfProb = 0
    for i  in range(len(newListOfProb)):
        sumOfProb = sumOfProb + newListOfProb[i]

    for i in range(len(newListOfProb)):
        newListOfProb[i] = newListOfProb[i]/sumOfProb

    return newListOfProb

"""
    This function returns T*W[x], it takes as inputs: x, W, gamma, rewards for each state, rewards for wins, loses, ties.
"""
def get_TstarWofX( X, W, gamma, rewardInfo, validStateIndices, winReward, tieReward, games, boardSize, agentSide):

    #get the valid_actions for agent
    valid_actions = get_listOfActions( boardSize, X , agentSide)
    
    #initialize W_bar to 0
    W_bar = []
    for i in range(len(validStateIndices)):
        W_bar.append(0)

    #This array contains R_bar[x,a] + Pa[x..]*W[x] for all actions a.
    maxR = []

    for action in range(len(valid_actions)):
        
        maxR.append(0)

        #copy the contents of X.
        next_state = []
        for m in range(boardSize):
            k = []
            for j in range(boardSize):
                k.append(X[m][j])
            next_state.append(k)
        
        #Append the action on the state
        if( agentSide == 'X' ):
            next_state[valid_actions[action][0]][valid_actions[action][1]] = 1
        else:
            next_state[valid_actions[action][0]][valid_actions[action][1]] = 2

        #if x wins after performing the action, return then directly return winReward, if it is a tie return tieReward
        game_over = games[getIndex(boardSize, next_state)]
        if( game_over[0] == True and game_over[2] == False ):
            maxR[action] = winReward + gamma*W[0]             
            continue
        elif( game_over[0] == True and game_over[2] == True ):
            maxR[action] = tieReward + gamma*W[0] 
            continue

        n = getIndex(boardSize, next_state) 
        if( agentSide == 'X' ):
            opponentSide = 'O'
        else:
            opponentSide = 'X'

        #get the list of probabilities of opponent for the state( state resulted after agent's action)
        opponentFilename = 'opponent'+str(boardSize)+opponentSide+'.txt'
        listOfProb = get_int_list(linecache.getline(opponentFilename, n+1))
        newListOfProb = getNormalizedProb(listOfProb)

        maxR_bar = 0
        maxPW_bar = 0
        for i in range(9):
            if( newListOfProb[i] != 0 ):

                #copy the state
                sampleState = []
                for m in range(boardSize):
                    k = []
                    for j in range(boardSize):
                        k.append(next_state[m][j])
                    sampleState.append(k)

                #get the state after opponent makes a move
                if( agentSide == 'X' ):
                    sampleState[i//boardSize][ i%boardSize] = 2
                else:
                    sampleState[i//boardSize][ i%boardSize] = 1

                n = getIndex(boardSize, sampleState)
                
                #caluculate the sum
                maxR_bar = maxR_bar + rewardInfo[n]*newListOfProb[i]
                k = 0
                b = False

                #update the maxPW_bar
                for j in range(len(validStateIndices)):
                    if( validStateIndices[j] == n ):
                        maxPW_bar = maxPW_bar + newListOfProb[i]*W[j]
                        b = True
                        break

        #append the value to maxR array
        maxR[action] = maxR_bar + gamma*maxPW_bar

    #find the maximum value among all maxR's and return
    result = maxR[0]
    for i in range(len(maxR)):
        if( maxR[i] > result ):
            result = maxR[i]
    return result

"""
    This function takes W and returns W_bar by performing T* on W.
"""
def getW_bar( W , rewardInfo, validStateIndices, winReward, tieReward, games, boardSize, agentSide, gamma):
    Wbar = []
    n = len(validStateIndices)

    for i in range( len(validStateIndices)):
        
        #get Wbar[x] for each state and append it to Wbar
        Wbar.append(get_TstarWofX( getConfig(validStateIndices[i], boardSize), W , gamma, rewardInfo, validStateIndices, winReward, tieReward, games, boardSize, agentSide))

        #progress bar
        if( i%200 == 0 ):
            j = (i+1)/n
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('#'*int(20*j), 100*j))
            sys.stdout.flush()

    return Wbar

"""
    This function performs the Value Iterations for n times and return the list of Value functions obtained at each iteration
"""
def iterateForVstar(n, winReward, loseReward, tieReward, games, boardSize, agentSide, gamma):
    l = []
    W = []
    validStateIndices = getValidStateIndices(agentSide)

    #initialize the first W
    for i in range(len(validStateIndices)):
        #W.append(random.randint(100, 500))
        W.append(0)

    l.append(W)
    for i  in range(n):

        #progress bar
        sys.stdout.write('\r')
        s = "computing V_stars iteration "+ str(i+1) + "        "
        sys.stdout.write(s)
        sys.stdout.flush()
        print()

        #get next W'a by repeatedly applying previous W
        l.append(getW_bar( l[i] , getRewardInfo(boardSize, winReward, loseReward, tieReward, games, agentSide), validStateIndices, winReward, tieReward, games, boardSize, agentSide, gamma)) 

    return l

"""
    This function extracts the policies from the list of Value functions from iterateForVstar and returns them
"""
def getPoliciesValueIteration(n, winReward, loseReward, tieReward, boardSize, agentSide, gamma):
    games = getGames(boardSize)

    #get the list of V's
    V_stars = iterateForVstar(n, winReward, loseReward, tieReward, games, boardSize, agentSide, gamma)
    policies = []
    
    #append the arbitrary policy as the first policy( this is just to run the game for an arbitrary policy for knowing the 
    #performance of an arbitrary policy
    policies.append(getPolicyVector(boardSize, agentSide))
    indices = getValidStateIndices(agentSide)

    for i in range(len(V_stars)):

        #progress bar
        sys.stdout.write('\r')
        s = "computing Pi Star for V_star "+ str(i) + "        "
        sys.stdout.write(s)
        sys.stdout.flush()
        print()

        #get pistar for each V.
        pistar = getPiStar(V_stars[i], getRewardInfo(boardSize, winReward, loseReward, tieReward, games, agentSide), getValidStateIndices(agentSide), winReward, tieReward, games, boardSize, agentSide, gamma)

        #make a policy from that piStar( since pistar has only 2123 states and policy has all states, we need to append 0 to those
        #redundant states
        policy = []
        for i in range(3**(boardSize*boardSize)):
            policy.append(0) 

        for i in range(len(pistar)):
            policy[indices[i]] = pistar[i]

        policies.append(policy)

    return policies

"""
    This function extracts pistar from a single V
"""
def getPiStar(V_Star, rewardInfo, validStateIndices, winReward, tieReward, games, boardSize, agentSide, gamma):
    pistar = []
    n = len(validStateIndices)
    for i in range(len(validStateIndices)):

        #progress bar
        if( i % 200 == 0 ):
            j = (i+1)/n
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('#'*int(20*j), 100*j))
            sys.stdout.flush()
        
        #get the pistar[x] for each state and append it.
        pistar.append(getPiStarX(getConfig(validStateIndices[i], boardSize), V_Star, rewardInfo, validStateIndices, winReward, tieReward, games, boardSize, agentSide, gamma))

    return pistar

"""
    This function returns pistar[x], given a pi
"""
def getPiStarX(X, W, rewardInfo, validStateIndices, winReward, tieReward, games, boardSize, agentSide, gamma):
    
    valid_actions = get_listOfActions( boardSize, X , agentSide)
    W_bar = []
    for i in range(len(validStateIndices)):
        W_bar.append(0)

    maxR = []
    for action in range(len(valid_actions)):
        maxR.append(0)

        #get the configuration for the current state
        next_state = []
        for m in range(boardSize):
            k = []
            for j in range(boardSize):
                k.append(X[m][j])
            next_state.append(k)
        
        #perform the agents action
        if( agentSide == 'X' ):
            next_state[valid_actions[action][0]][valid_actions[action][1]] = 1
        else:
            next_state[valid_actions[action][0]][valid_actions[action][1]] = 2
        
        #if x wins, return winReward, else if it is a tie, return tie reward
        game_over = games[getIndex(boardSize, next_state)]
        if( game_over[0] == True and game_over[2] == False ):
            maxR[action] = winReward + gamma*W[0]
            continue
        elif( game_over[0] == True and game_over[2] == True ):
            maxR[action] = tieReward + gamma*W[0]
            continue
        
        n = getIndex(boardSize, next_state) 
        if( agentSide == 'X' ):
            opponentSide = 'O'
        else:
            opponentSide = 'X'

        #get the probabilities for the current state( state resulted after agent performs an action )
        opponentFilename = 'opponent'+str(boardSize)+opponentSide+'.txt'
        listOfProb = get_int_list(linecache.getline(opponentFilename, n+1))
        newListOfProb = getNormalizedProb(listOfProb)
            
        maxR_bar = 0
        maxPW_bar = 0
        for i in range(9):
            if( newListOfProb[i] != 0 ):

                #copy the state
                sampleState = []
                for m in range(boardSize):
                    k = []
                    for j in range(boardSize):
                        k.append(next_state[m][j])
                    sampleState.append(k)

                #get the state after opponent makes a move
                if( agentSide == 'X' ):
                    sampleState[i//boardSize][ i%boardSize] = 2
                else:
                    sampleState[i//boardSize][ i%boardSize] = 1
                n = getIndex(boardSize, sampleState)

                #calculate the sum
                maxR_bar = maxR_bar + rewardInfo[n]*newListOfProb[i]
                k = 0
                b = False

                #update the maxPW-bar
                for j in range(len(validStateIndices)):
                    if( validStateIndices[j] == n ):
                        maxPW_bar = maxPW_bar + newListOfProb[i]*W[j]
                        b = True
                        break
        #append the value to maxR array
        maxR[action] = maxR_bar + gamma*maxPW_bar

    #find the action that resulted in the maximum R, and return it
    result = maxR[0]
    for i in range(len(maxR)):
        if( maxR[i] > result ):
            result = maxR[i]
    for i in range(len(maxR)):
        if( maxR[i] == result ):
            return valid_actions[i]
