from helper import * 
from opponent import *
from agent import *
from ValueIteration import *
from PolicyIteration import *
from Q_learning import getPIStarQ_learning
from SARSA import getPIStarSARSA
import linecache
import os.path
import random

"""
    This function plays a single game/episode, until the end( win/draw ).
    After playing the game, it returns 0 if it is a tie, 1 if agent wins,
    and -1 if opponent wins. The agent policy is the input.
"""
def getWinner(pistar, opponentSide):
    
    #Compute the Agent Side from opponent Side
    if( opponentSide == 'O' ):
        agentSide = 'X'
    else:
        agentSide = 'O'

    #randomly select X or O to start the Game
    temp = random.randint(0, 1)
    if temp == 0:
        player = 'O'
    else:
        player = 'X'
    
    #set the board
    gameover = 0
    config = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]
    
    #play until game completes
    while( gameover == 0 ):


        if( player == agentSide ):
            
            #get the agents move and change the board
            agentmove = pistar[getIndex(3, config)]
            if( player == 'X' ):
                config[agentmove[0]][agentmove[1]] = 1
            else:
                config[agentmove[0]][agentmove[1]] = -1
            
            #if game is over return the winner
            game_over = get_game_over(3, config)
            if( game_over[0] == True ):
                return game_over[1]
            
            #if not change the player so that opponent makes next move
            player = opponentSide
            continue

        if( player == opponentSide ):

            #get the opponent move the change the board
            opponentmove = getOpponentMove( config, 3, opponentSide )
            if( player == 'O' ):
                config[opponentmove[0]][opponentmove[1]] = -1
            else:
                config[opponentmove[0]][opponentmove[1]] = 1

            #if game is over return the winner
            game_over = get_game_over(3, config)
            if( game_over[0] == True ):
                return game_over[1]

            #agent makes next move
            player = agentSide
    
"""
    This function plays n games with the given policy and outputs 
    a tuple(no of times agent wins, no of times agent loses, no of ties).
"""
def playGame(n, policy, opponentSide):

    #initialize counts
    countX = 0
    countO = 0
    countD = 0
    startX = 0
    startO = 0
    for i in range(0, n):

        #get the winner
        winner = getWinner(policy, opponentSide)

        #progress bar
        if( i%1000 == 0 ):
            j = (i+1)/n
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%  " % ('#'*int(20*j), 100*j))
            sys.stdout.flush()

        #increment appropriate values
        if( winner == 1 ):
            countX = countX + 1
        elif( winner == 2 ):
            countO = countO + 1
        elif( winner == 0 ):
            countD = countD + 1

    sys.stdout.write('\r')
    sys.stdout.flush()

    #return the values (agent wins, agent loses, ties)
    if( opponentSide == 'O' ):
        return (countX, countO, countD)
    else:
        return (countO, countX, countD)


"""
    This is the Main function of the file.
"""
green = "\033[92m"
blue = "\033[94m"
end = "\033[0m"
boardSize = 3 


print(green+"Enter 1 if agent plays X or 0 if agent plays O"+end)
choice = int(input())
if( choice == 0 ):
    agentSide = 'O'
    opponentSide = 'X'
else:
    agentSide = 'X'
    opponentSide = 'O'

print(green+"Enter Reward for winning"+end)
winReward = int(input())
print(green+"Enter Reward for loosing"+end)
loseReward = int(input())
print(green+"Enter Reward for a Tie"+end)
tieReward = int(input())
print(green+"Enter 1 to create a new opponent and 0 to use existing opponent"+end)
choice = int(input())
if choice == 1:
    createOpponent(boardSize, True, opponentSide)
else:
    createOpponent(boardSize, False, opponentSide)
print(green+"Enter 1 to create a new policy and 0 to use existing policy" + end)
choice = int(input())
if choice == 1:
    createPolicy( boardSize, True, agentSide)
else:
    createPolicy(boardSize, False, agentSide)



print(green + "Enter 0 to use Arbitrary Policy,\n1 to use ValueIteration,\n2 to use PolicyIteration,\n3 to use Q-Learning,\n4 to use SARSA"+ end)
choice = int(input())
if( choice == 0 ):
    print(green + "Enter the Number of Games to Play"+end)
    n = int(input())
    res = playGame(n, getPolicyVector(boardSize, agentSide), opponentSide)
    print(blue+"Final Results                 ")
    print("Agent Wins\tAgent Loses\tTies")
    print(res[0],'\t\t', res[1], '\t\t', res[2])

elif( choice == 1 ):
    print(green+"Enter the Number of Games to Play"+end)
    n = int(input())
    print(green+"Enter Number of Iterations"+end)
    itr = int(input())
    print(green+"Enter Gamma"+end)
    gamma = float(input())
    policies = getPoliciesValueIteration(itr, winReward, loseReward, tieReward, boardSize, agentSide, gamma)

    string = '\t\tAgent Wins\tAgent Loses\tTies\n'
    string += 'Arb Policy'
    for i in range(len(policies)):
        res = playGame(n, policies[i], opponentSide)
        if( i != 0 ):
            string += 'Iteration '+str(i-1)
        string += '\t' + str(res[0]) + '\t\t' + str(res[1]) + '\t\t' + str(res[2]) + '\n'

    print(blue+"Final Results          "+end)
    print(blue+string+end)
    print(green+"Enter 1 if you want to save this policy as an opponent\n0 to exit with saving"+end)
    choice = int(input())
    if( choice == 1 ):
        print(green+"Enter the weight you want to give the policy in the probability transition function (0-100)"+end)
        choice = int(input())
        generateOpponent( policies[len(policies)-1], agentSide, choice)

elif( choice == 2 ):
    print(green+"Enter the Number of Games to Play"+end)
    n = int(input())
    print(green+"Enter the Number of Iterations"+end)
    itr = int(input())
    print(green+"Enter Gamma"+end)
    gamma = float(input())
    policies = getPoliciesPolicyIteration(itr, winReward, loseReward, tieReward, agentSide, gamma)
    
    string = '\t\tAgent Wins\tAgent Loses\tTies\n'
    string += 'Arb Policy'
    for i in range(len(policies)):
        res = playGame(n, policies[i], opponentSide)
        if( i != 0 ):
            string += 'Iteration '+str(i-1)
        string += '\t' + str(res[0]) + '\t\t' + str(res[1]) + '\t\t' + str(res[2]) + '\n'

    print(blue+"Final Results:                 ")
    print(blue+string+end)
    print(green+"Enter 1 if you want to save this policy as an opponent\n0 to exit with saving"+end)
    choice = int(input())
    if( choice == 1 ):
        print(green+"Enter the weight you want to give the policy in the probability transition function (0-100)"+end)
        choice = int(input())
        generateOpponent( policies[len(policies)-1], agentSide, choice)

elif( choice == 3 ):
    print(green + "Enter the Number of Games to Play"+end)
    n = int(input())
    print(green+"Enter Number of Episodes of trajectory you want to train"+end)
    itr = int(input())
    policy = getPIStarQ_learning( itr, winReward, loseReward, tieReward, agentSide)
    res = []
    res.append(playGame(n, getPolicyVector(3, agentSide), opponentSide))
    res.append(playGame(n, policy, opponentSide))

    print(blue+"Final Results                 ")
    print("\t\tAgent Wins\tAgent Loses\tTies")
    print('Arb Policy\t',res[0][0],'\t\t', res[0][1], '\t\t', res[0][2])
    print('Q-learned\t',res[1][0],'\t\t', res[1][1], '\t\t', res[1][2])
    print(green+"Enter 1 if you want to save this policy as an opponent\n0 to exit with saving"+end)
    choice = int(input())
    if( choice == 1 ):
        print(green+"Enter the weight you want to give the policy in the probability transition function (0-100)"+end)
        choice = int(input())
        generateOpponent( policy, agentSide, choice)

elif( choice == 4 ):
    print(green + "Enter the Number of Games to Play"+end)
    n = int(input())
    print(green+"Enter Number of Episodes of trajectory you want to train"+end)
    itr = int(input())
    policy = getPIStarSARSA( itr, winReward, loseReward, tieReward, agentSide)
    res = []
    res.append(playGame(n, getPolicyVector(3, agentSide), opponentSide))
    res.append(playGame(n, policy, opponentSide))

    print(blue+"Final Results                 ")
    print("\t\tAgent Wins\tAgent Loses\tTies")
    print('Arb Policy\t', res[0][0],'\t\t', res[0][1], '\t\t', res[0][2])
    print('SARSA\t\t', res[1][0],'\t\t', res[1][1], '\t\t', res[1][2])
    print(green+"Enter 1 if you want to save this policy as an opponent\n0 to exit with saving"+end)
    choice = int(input())
    if( choice == 1 ):
        print(green+"Enter the weight you want to give the policy in the probability transition function (0-100)"+end)
        choice = int(input())
        generateOpponent( policy, agentSide, choice)
