"""
    This file contains the functions to create an agent's policy, and also to get the move of the agent according to the policy for a given state.
"""
from helper import *
import linecache
import os.path
import random

#Given a string from the line, parse and convert it into a coordinate list.
def getCoordinate( line ):
    line = line[:-1]        #remove '\n'
    line = line.split()
    for i in range(len(line)):
        line[i] = int(line[i])      #parse into int

    return line             #return the list

#This function returns the agents move for a given state according to the policy 
def getAgentMove( configuration, boardSize):
    config = []
    for i in range(boardSize):
        k = []
        for j in range(boardSize):
            if configuration[i][j] == -1:           #create a new copy of the state with -1 replaced with 2
                k.append(2)
            else:
                k.append(configuration[i][j])
        config.append(k)
    
    n = getIndex(boardSize, config)         #calculate the index of the state

    policyFilename = 'policy'+str(boardSize)+'.txt'
    coordinate = getCoordinate(linecache.getline(policyFilename, n+1))      #get the corresponding coordinate from the policy file
    return coordinate           #return the coordinate

#This function returns an arbitrary coordinate out of all the legal action coordinates, so we will achieving an arbitrary policy
def getPolicy( index, boardSize ):
    actions = get_listOfActions( boardSize, getConfig(index, boardSize) , 'X' )         #get all the possible legal actions
    if len(actions) == 0:
        return []
    pos = random.randint(0, len(actions)-1)                 #randomly select one action from the list of actions
    return actions[pos]         #return that particular action


#This function creates an arbitrary policy if not present already
def createPolicy( boardSize ):
    rows = 3**(boardSize*boardSize)
    policyFilename = 'policy'+str(boardSize)+'.txt'

    if os.path.exists(policyFilename) == False:     #if the file/policy doesn't already exists
        file = open(policyFilename, 'w')        #create a new file
        r = rows//100
        print('Please wait while the new policy is being created')
        for i in range(rows):
            coordinate = getPolicy( i, boardSize )      #get an arbitrary coordinate as policy for each state
            string = ""
            if len(coordinate) != 0:
                string = str(coordinate[0])+ ' ' + str(coordinate[1])
            string = string + '\n'
            if i % r == 0 and i != 0:
                    print(str(i//r)+'% completed')
            file.write(string)      #insert the coordinate into the file
        print('Created new policy')

    else:
        print('Using existing policy')
