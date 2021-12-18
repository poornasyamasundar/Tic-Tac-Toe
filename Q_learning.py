from helper import *
import sys
from opponent import *
import linecache
import os.path
import random

#update function for the Q table
def updateQ( frequency, Q, beforeState, currentState, action, Rt, validStateIndices, itr ):
    gamma = 0.1
    temp = getIndex(3,beforeState)
    for i in range(len(validStateIndices)):
        if( temp == validStateIndices[i] ):
            St = i
            break;

    frequency[St] += 1
    temp = getIndex(3,currentState)
    for i in range(len(validStateIndices)):
        if( temp == validStateIndices[i] ):
            Stplus1 = i
            break;

    At = action[0]*3 + action[1]
    alpha = 1/(itr+1)
    listOfActions = get_listOfActions( 3, currentState, 'X' )
    maxQt = -10000
    for i in range(len(listOfActions)):
        a = listOfActions[i][0]*3 + listOfActions[i][1]
        if( Q[Stplus1][a] > maxQt ):
            maxQt = Q[Stplus1][a]

    Q[St][At] = Q[St][At] + alpha*(Rt + (gamma*maxQt) - Q[St][At])
    return


#get the Policy throught Q learning
def getPIStarQ_learning(n, winReward, loseReward, tieReward, agentSide):
    validStateIndices = getValidStateIndices(agentSide)
    noOfGames = 0
    noOfXWins = 0
    noOfOWins = 0
    noOfDraws = 0
    frequency = []
    if( agentSide == 'X' ):
        opponentSide = 'O'
    else:
        opponentSide = 'X'
    for i in range(len(validStateIndices)):
        frequency.append(0)

    gamma = 0.1
    Q = []
    for i in range (len(validStateIndices)):
        Q.append([0, 0, 0, 0, 0, 0, 0, 0, 0])       #initially the Q table is empty
    currentState = getConfig(0, 3)
    gameover = 1
    for i in range(n):          #play games for n times
        choice = random.randint(0, 1)       #choose which player should play first
        currentState = getConfig(0, 3)
        if( choice == 0 ):
            while(True):
                listOfActions = get_listOfActions( 3, currentState, agentSide )
                action = listOfActions[random.randint(0, len(listOfActions)-1)]
                stateBeforeX = []
                for j in range(3):
                    l = []
                    for k in range(3):
                        l.append(currentState[j][k])
                    stateBeforeX.append(l)

                if( agentSide == 'X' ):
                    currentState[action[0]][action[1]] = 1
                else:
                    currentState[action[0]][action[1]] = -1
                if( get_game_over(3, currentState)[0] == True and get_game_over(3, currentState)[2] == False ):
                    currentState = getConfig(0, 3)
                    updateQ( frequency, Q, stateBeforeX, currentState, action, winReward, validStateIndices, i )
                    noOfXWins += 1
                    break

                if( get_game_over(3, currentState)[0] == True and get_game_over(3, currentState)[2] == True ):
                    currentState = getConfig(0, 3)
                    updateQ( frequency, Q, stateBeforeX, currentState, action, tieReward, validStateIndices, i)
                    noOfDraws += 1
                    break
                
                opponentmove = getOpponentMove( currentState, 3, opponentSide )
                if( opponentSide == 'O' ):
                    currentState[opponentmove[0]][opponentmove[1]] = -1
                else:
                    currentState[opponentmove[0]][opponentmove[1]] = 1

                if( get_game_over(3, currentState)[0] == True and get_game_over(3, currentState)[2] == False ):
                    currentState = getConfig(0, 3)
                    updateQ( frequency, Q, stateBeforeX, currentState, action, loseReward, validStateIndices, i)
                    noOfOWins += 1
                    break

                if( get_game_over(3, currentState)[0] == True and get_game_over(3, currentState)[2] == True ):
                    currentState = getConfig(0, 3)
                    updateQ( frequency, Q, stateBeforeX, currentState, action, tieReward, validStateIndices, i)
                    noOfDraws += 1
                    break

                updateQ( frequency, Q, stateBeforeX, currentState, action, 0, validStateIndices, i)
        else:       #else O plays first
            opponentmove = getOpponentMove( currentState, 3, opponentSide)
            if( opponentSide == 'O' ):
                currentState[opponentmove[0]][opponentmove[1]] = -1
            else:
                currentState[opponentmove[0]][opponentmove[1]] = 1
            
            while(True):
                listOfActions = get_listOfActions( 3, currentState, agentSide )
                action = listOfActions[random.randint(0, len(listOfActions)-1)]
                stateBeforeX = []
                for j in range(3):
                    l = []
                    for k in range(3):
                        l.append(currentState[j][k])
                    stateBeforeX.append(l)
                if( agentSide == 'X' ):
                    currentState[action[0]][action[1]] = 1
                else:
                    currentState[action[0]][action[1]] = -1
                if( get_game_over(3, currentState)[0] == True and get_game_over(3, currentState)[2] == False ):
                    currentState = getConfig(0, 3)
                    updateQ( frequency, Q, stateBeforeX, currentState, action, winReward, validStateIndices, i )
                    noOfXWins += 1
                    break

                if( get_game_over(3, currentState)[0] == True and get_game_over(3, currentState)[2] == True ):
                    currentState = getConfig(0, 3)
                    updateQ( frequency, Q, stateBeforeX, currentState, action, tieReward, validStateIndices, i)
                    noOfDraws += 1
                    break

                opponentmove = getOpponentMove( currentState, 3, opponentSide)
                if( opponentSide == 'O' ):
                    currentState[opponentmove[0]][opponentmove[1]] = -1
                else:
                    currentState[opponentmove[0]][opponentmove[1]] = 1

                if( get_game_over(3, currentState)[0] == True and get_game_over(3, currentState)[2] == False ):
                    currentState = getConfig(0, 3)
                    updateQ( frequency, Q, stateBeforeX, currentState, action, loseReward, validStateIndices, i)
                    noOfOWins += 1
                    break

                if( get_game_over(3, currentState)[0] == True and get_game_over(3, currentState)[2] == True ):
                    currentState = getConfig(0, 3)
                    updateQ( frequency, Q, stateBeforeX, currentState, action, tieReward, validStateIndices, i)
                    noOfDraws += 1
                    break

                updateQ( frequency, Q, stateBeforeX, currentState, action, 0, validStateIndices, i)
        if( i%100 == 0 ):
            j = (i+1)/n
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('#'*int(20*j), 100*j))
            sys.stdout.flush()

    sys.stdout.write('\r')
    sys.stdout.flush()
    print("Training Episodes statistics")
    print("No of games Agent won = ", noOfXWins)
    print("No of games Opponent won = ", noOfOWins)
    print("No of games Tie = ", noOfDraws)
    
    #Extract policy from the final Q_table
    Pi = []
    for i in range(len(Q)):
        m = -10000
        ma = 0
        listOfActions = get_listOfActions( 3, getConfig(validStateIndices[i], 3), agentSide )
        for j in range(len(listOfActions)):
            a = listOfActions[j][0]*3 + listOfActions[j][1]
            if( Q[i][a] > m ):
                m = Q[i][a]
                ma = a;
        Pi.append([ma//3, ma%3])

    policy = []
    for j in range(19683):
        policy.append(0)
    for j in range(len(Pi)):
        policy[validStateIndices[j]] = Pi[j]
    
    return policy

