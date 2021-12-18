"""
    This file contains the functions to create an agent's policy, and also to get the move of the agent according to the policy for a given state.
"""
from helper import *
from opponent import *
import linecache
import os.path
import random
import sys

#Given a string from the line, parse and convert it into a coordinate list.
def getCoordinate( line ):
    line = line[:-1]        #remove '\n'
    line = line.split()
    for i in range(len(line)):
        line[i] = int(line[i])      #parse into int

    return line             #return the list

#This function returns an arbitrary coordinate out of all the legal action coordinates, so we will achieving an arbitrary policy
def getPolicy( index, boardSize, agentSide ):
    actions = get_listOfActions( boardSize, getConfig(index, boardSize) , agentSide )         #get all the possible legal actions
    if len(actions) == 0:
        return []
    pos = random.randint(0, len(actions)-1)                 #randomly select one action from the list of actions
    return actions[pos]         #return that particular action


def createPolicy( boardSize, choice, agentSide ):
    rows = 3**(boardSize*boardSize)
    policyFilename = 'policy'+str(boardSize)+agentSide+'.txt'

    if os.path.exists(policyFilename) == False:
        choice = True
    if choice == True:
        file = open(policyFilename, 'w')        #create a new file
        r = rows//100
        print('Please wait while the new policy is being created')
        n = rows
        for i in range(rows):
            coordinate = getPolicy( i, boardSize, agentSide )      #get an arbitrary coordinate as policy for each state
            string = ""
            if len(coordinate) != 0:
                string = str(coordinate[0])+ ' ' + str(coordinate[1])
            string = string + '\n'
            if( i % 1000 == 0 ):
                j = (i+1)/n
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" % ('#'*int(20*j), 100*j))
                sys.stdout.flush()
            file.write(string)      #insert the coordinate into the file
        sys.stdout.write('\r')
        sys.stdout.flush()
        print("\033[94m" + 'Created new policy for agent playing '+agentSide+'                 ' + "\033[0m" )   
    else:
        print("\033[94m" + 'Using existing policy for agent playing ' + agentSide+"\033[0m" )   

