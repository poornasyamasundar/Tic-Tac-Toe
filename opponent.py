from helper import *
import linecache
import os.path
import random
import sys

"""
    This file contains the functions to create an arbitrary opponent with an arbitrary probability transistion function.
    The need for a probability transition function is to create an environment which acts according to a probability function
    To achieve this we do the following:
        Once the agent performs a move on a given board configuration, the next state the agent will face is the board configuration with an additional O' in the table placed by the opponent.
        Now the resulting state can be made probabilistic by imposing a probability function on the place in which the opponent places his 'O'
        i.e, we can tell the opponent to place 'O' at position 1 for 10% of time, in position 2 for 30% of time and so on, where position 1, 2, 3, are the positions in which the opponent can place an
        'O'.
        So what we do is that for each state in the state space we first calculate all the positions in which the opponent can place an 'O', assign probabilites on the positions. So each time the state
        is occured the opponent chooses the place to put 'O' according to this probability.

        Since the state space is large and we need the opponent to be consistent, we store the probability table in a file and use it.
"""
#This function given a string ("1 2 3 4 5\n") of space integers in a file, converts it into a list 
def get_int_list(line):
    line= line[:-1]         #remove '\n'
    line = line.split()     #use split() to create a list
    for i in range(len(line)):
        line[i] = int(line[i])      #for each list item convert it to int from string

    return line         #return the list [1, 2, 3, 4, 5]


#This function takes a state and then outputs the probabilites with which the 'O' should be placed in each cell
def get_probability( boardSize, configuration, opponentSide ):
    actions = get_listOfActions( boardSize, configuration, opponentSide )    #first get the list of all positions in which an 'O' can be placed
    if len(actions) == 0:       #if there are no possible actions then return emptry array
        return []
    probabilities = []
    for i in range(len(actions)):       
        probabilities.append(0)         #initially assign a probability of zero to each cell

    for i in range(len(actions)):
        #probabilities[random.randint(0,len(actions)-1)] += 1        # to arbitrarily divide the probablity 1 among all the possibel actions, select randomly each action, the number of times an action is
        probabilities[i] = random.randint(100, 200)
        #probabilities[i] = 1                                                                                           # the corresponding probability with which that action will be choosed
    #probabilities[random.randint(0, len(actions)-1)] = 1
    listOfProb = [0 for i in range(boardSize*boardSize)]
    s = 0
    for i in range(len(actions)):
        s = s + probabilities[i]
        if probabilities[i] != 0:
            listOfProb[ ( actions[i][0] * boardSize ) + actions[i][1] ] = s     # store the probabilites in cumulative order

    return listOfProb           #return the probabilites

# Given a configuration this function returns the position in which the 'O' should be placed
def getOpponentMove( configuration, boardSize, opponentSide ):
    config = []
    for i in range(boardSize):
        k = []
        for j in range(boardSize):
            if configuration[i][j] == -1:           # create a new 2d array with -1 replaced by 2 
                k.append(2)
            else:
                k.append(configuration[i][j])
        config.append(k)
    
    n = getIndex(boardSize, config)         #get the index of the configuration
    opponentFilename = 'opponent'+str(boardSize)+opponentSide+'.txt'     #compute the file name in which the probabilites are placed
    listOfProb = get_int_list(linecache.getline(opponentFilename, n+1))     #get the corresponding line and convert it into a list

    maximum = 0
    for i in range( len(listOfProb) ):
        if listOfProb[i] > maximum:             #find the max in the list i.e, the cumulative sum of all the probabilites assigned
            maximum = listOfProb[i]         

    k = random.randint(1, maximum)      #randomly select a number from the cumulative sum
    for i in range(len(listOfProb)):
        if k <= listOfProb[i]:                      #if the random number is less than a any of the cummulative probabilites assigned then select that move and return the coordinates
            return [i//boardSize, i%boardSize]
    
    return []

#This function creates an opponent if it is not present already
def createOpponent( boardSize, choice, opponentSide ):
    rows = 3**(boardSize*boardSize)
    opponentFilename = 'opponent'+str(boardSize)+opponentSide+'.txt'

    n = rows
    if os.path.exists(opponentFilename) == False: 
        choice = True
    if choice == True:
        file = open(opponentFilename, 'w')
        print("Please wait while the opponent is being created")
        for i in range(rows):
            file.write(getString_from_intlist(get_probability( boardSize, getConfig(i, boardSize), opponentSide)))        #for each index, find the probabilites and push into the file
            if( i % 1000 == 0 ):
                j = (i+1)/n
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" % ('#'*int(20*j), 100*j))
                sys.stdout.flush()
        sys.stdout.write('\r')
        sys.stdout.flush()
        print("\033[94m" + 'Created new opponent for playing '+opponentSide+'               ' + "\033[0m" )   
    else:                       #if the file already exists return
        print("\033[94m" + 'Using existing opponent for playing '+opponentSide + "\033[0m" )   

