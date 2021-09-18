#This file contains helper functions to manage states and actions

#this function assigns an unique index for each configuration
def getIndex( boardSize, configuration ):
    index = 0
    for i in range(boardSize):
        for j in range(boardSize):
            index = index*3                 #calculate the index by using base-3 number system i.e, assume the list of cells as a base-3 number and convert into a decimal system and store it
            index = index + configuration[i][j]

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
            elif configuration[i][j] == 2:
                counto += 1

    if player == 'X':
        if countx != counto:        #if the number X's and O's are not equal then it is not X's turn, so return an empty list which means on action is legal
            return []
                    
    elif player == 'O':
        if countx != counto+1:      #similarly if the number of X's is not 1 more than number of O's then it is not O's turn
            return []

    actions = []
    for i in range(boardSize):
        for j in range(boardSize):
            if configuration[i][j] == 0:            #if the cell is empty then add it to the list of legal actions
                actions.append([i,j])
                    
    return actions              #return the list of actions



