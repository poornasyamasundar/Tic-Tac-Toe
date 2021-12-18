import linecache
import os.path
import random
import sys
#This file contains helper functions to manage states and actions

#given a board configuration this returns, if it is a tie, win, lose or intermediate

def getString_from_intlist(l):
    line = ""
    for i in range(len(l)):
        line = line+str(l[i])+' '       #add the integer and then a space
    line = line+ '\n'                   #add a new line
    return line                     #return the string("1 2 3 4 5\n")


def get_game_over(board_layout,configuration):
    game_over = False
    winner = 0

    markers = []
    for i in range(board_layout):
        k = []
        for j in range(board_layout):
            if configuration[i][j] == 2:           #create a new copy of the state with -1 replaced with 2
                k.append(-1)
            else:
                k.append(configuration[i][j])
        markers.append(k)

    #check columns
    for i in range(board_layout):
        colsum = 0
        for j in range(board_layout):
            colsum = colsum+markers[j][i]
        if colsum == board_layout:
            winner = 1
            game_over = True
        if colsum == -1*board_layout:
            winner = 2
            game_over = True

    #check rows
    for i in range(board_layout):
        rowsum = 0
        for j in range(board_layout):
            rowsum = rowsum+markers[i][j]

        if rowsum == board_layout:
            winner = 1
            game_over = True
        if rowsum == -1*board_layout:
            winner = 2
            game_over = True

    #check diagnols
    main_diagnol_sum = 0
    for i in range(board_layout):
        main_diagnol_sum = main_diagnol_sum + markers[i][i]

    if main_diagnol_sum == board_layout:
        winner = 1
        game_over = True
    if main_diagnol_sum == -1*board_layout:
        winner = 2
        game_over = True

    other_diagnol_sum = 0
    startrow = board_layout -1
    startcol = 0
    for i in range(board_layout):
        other_diagnol_sum = other_diagnol_sum + markers[startrow][startcol]
        startrow = startrow - 1
        startcol = startcol + 1

        if other_diagnol_sum == board_layout:
            winner = 1
            game_over = True
        if other_diagnol_sum == -1*board_layout:
            winner = 2
            game_over = True

    #check Ties
    tie = False
    if game_over == False:
        tie = True
        for i in range(board_layout):
            for j in range(board_layout):
                if markers[i][j] == 0:
                    tie = False
        #print("tie is ", tie)
        #if it is a tie, then call game over and set winner to 0(no one)
        if tie == True:
            game_over = True
            winner = 0
    return (game_over, winner, tie)


def getGames(board_layout):
    games = []
    for i in range(3**(board_layout*board_layout)):
        games.append(get_game_over(board_layout, getConfig( i, board_layout)))
    return games

#return the list of valid states
def getValidStates(agentSide):
    validStates = []
    for i in range(3**9):
        if( get_game_over(3, getConfig(i, 3))[0] == False ):
            if( len(get_listOfActions(3, getConfig(i, 3), agentSide)) != 0 ):
                validStates.append(1)
            else:
                validStates.append(0)
        else:
            validStates.append(0)

    return validStates

#return the indices of valid States
def getValidStateIndices(agentSide):
    validStates = getValidStates(agentSide)
    indices = []
    for i in range(len(validStates)):
        if( validStates[i] == 1 ):
            indices.append(i)
    return indices

#this function assigns an unique index for each configuration
def getIndex( boardSize, configuration ):
    index = 0
    for i in range(boardSize):
        for j in range(boardSize):
            if( configuration[i][j] == -1 ):
                configuration[i][j] = 2
            index = index*3                 #calculate the index by using base-3 number system i.e, assume the list of cells as a base-3 number and convert into a decimal system and store it
            index = index + configuration[i][j]
    
    for i in range(boardSize):
        for j in range(boardSize):
            if( configuration[i][j] == 2 ):
                configuration[i][j] = -1
    return index

#Given an index this function returns the configuration
def getConfig( index, boardSize ):
    configuration = []
    for i in range(boardSize):
        k = []
        for j in range(boardSize):
            k.insert(0, index % 3)          #this is reverse of the above function, given decimal number return the base-3 representation of the decimal number.
            index = index // 3
        configuration.insert(0, k)
    return configuration


#this function returns all the possible actions for a given player and for a given state, i.e. return all the positions where the player can place his symbol
def get_listOfActions( boardSize, configuration , player):
    countx = 0
    counto = 0
    for i in range(boardSize):
        for j in range(boardSize):
            if configuration[i][j] == 1:            #count the number of X's an O's in the state present already
                countx += 1
            elif configuration[i][j] == 2 or configuration[i][j] == -1 :
                counto += 1

    if player == 'X':
        if countx == counto or countx+1 == counto:
        #if countx+1 == counto:
            actions = []
            for i in range(boardSize):
                for j in range(boardSize):
                    if configuration[i][j] == 0:
                        actions.append([i,j])
            return actions
        else:
            return []
    if player == 'O':
        if countx == counto+1 or counto == countx:
        #if countx == counto:
            actions = []
            for i in range(boardSize):
                for j in range(boardSize):
                    if configuration[i][j] == 0:
                        actions.append([i,j])
            return actions
        else:
            return []
    return []

def getPolicyVector(boardSize, agentSide):
    policyFilename = 'policy'+str(boardSize)+agentSide+'.txt'
    policyVector = []
    for i in range(3**(boardSize*boardSize)):
        policyVector.append(getCoordinate(linecache.getline(policyFilename, i+1)))
    return policyVector

def getCoordinate( line ):
    line = line[:-1]        #remove '\n'
    line = line.split()
    for i in range(len(line)):
        line[i] = int(line[i])      #parse into int

    return line             #return the list

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

def generateOpponent( policy, opponentSide, percent ):

    boardSize = 3
    rows = 3**(boardSize*boardSize)
    opponentFilename = 'opponent'+str(boardSize)+opponentSide+'.txt'
    file = open(opponentFilename, 'w')
    print("Please wait while the opponent is being created")

    for i in range(rows):

        if( policy[i] == 0 ):
            file.write(getString_from_intlist(get_probability( boardSize, getConfig(i, boardSize), opponentSide)))
            continue

        probabilities = []
        actions = get_listOfActions( boardSize, getConfig(i, boardSize), opponentSide )

        if len(actions) != 0:
            for j in range(len(actions)):
                probabilities.append(0)

            index = 0
            for j in range(len(actions)):
                if( actions[j][0] != policy[i][0] or actions[j][1] != policy[i][1] ):
                    probabilities[j] = random.randint(100, 200)
                else:
                    index = j

            s = 0
            for j in range(len(actions)):
                s += probabilities[j]
            
            if( s != 0 ):
                probabilities[index] = (s//(100-percent))*percent
            else:
                probabilities[index] = 100

            s = 0
            listOfProb = []
            for j in range(boardSize*boardSize):
                listOfProb.append(0)
            for j in range(len(actions)):
                s = s + probabilities[j] 
                if probabilities[j] != 0:
                    listOfProb[ (actions[j][0]*3) + actions[j][1] ] = s
            file.write(getString_from_intlist(listOfProb))
        else:
            file.write(getString_from_intlist(probabilities))


        #progress bar
        if( i % 1000 == 0 ):
            j = (i+1)/rows
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('#'*int(20*j), 100*j))
            sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()
    
    print("\033[94m" + 'Successfully created opponent from the policy '+ "\033[0m" )   
