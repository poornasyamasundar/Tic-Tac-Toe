import pygame
from pygame.locals import *

pygame.init()

board_layout = 5
screen_height = board_layout*100
screen_width = board_layout*100
line_width = 6
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Tic Tac Toe '+str(board_layout)+' X '+str(board_layout))

#define colours
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

#define font
font = pygame.font.SysFont(None, 40)

#define variables
clicked = False
player = 1
pos = (0, 0)
markers = []
game_over = False
winner = 0

#setup a rectangle for "Play Again" Option
again_rect = Rect(screen_width//2 -80, screen_height//2, 160, 50)

#create empty n x n list to represent the grid ( n = board_layout )
markers = [[0 for i in range(board_layout)] for j in range(board_layout)]

def draw_board():
        bg = (255, 255, 210)
        grid = (50, 50, 50)
        screen.fill(bg)
        for x in range(1, board_layout):
                pygame.draw.line(screen, grid, (0, 100*x), (screen_width, 100*x), line_width)
                pygame.draw.line(screen, grid, (100*x, 0), (100*x, screen_height), line_width)


def draw_markers():
        y_pos = 0
        for i in range(board_layout):
                x_pos = 0
                for j in range(board_layout):
                        if markers[i][j] == 1:
                                pygame.draw.line(screen, red, (x_pos*100 + 15, y_pos*100+15), (x_pos*100+85, y_pos*100+85), line_width)
                                pygame.draw.line(screen, red, (x_pos * 100 + 85, y_pos * 100 + 15), (x_pos * 100 + 15, y_pos * 100 + 85), line_width)
                        if markers[i][j] == -1:
                                pygame.draw.circle(screen, green, (x_pos*100+50, y_pos*100+50), 38, line_width)
                        x_pos += 1
                y_pos += 1

def check_game_over():
        global game_over
        global winner

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
        if game_over == False:
                tie = True
                for i in range(board_layout):
                        for j in range(board_layout):
                                if markers[i][j] == 0:
                                        tie = False

                #if it is a tie, then call game over and set winner to 0(no one)
                if tie == True:
                        game_over = True
                        winner = 0

def draw_game_over(winner):

        if winner != 0:
                if winner == 1:
                        end_text = "Player X wins!"
                elif winner == 2:
                        end_text = "Player O wins!"
        elif winner == 0:
                end_text = "You have tied!"

        end_img = font.render(end_text, True, blue)
        pygame.draw.rect(screen, green, (screen_width//2 -100, screen_height // 2 - 60, 200, 50))
        screen.blit(end_img, (screen_width // 2 -100, screen_height // 2 - 50))

        again_text  = 'Play Again?'
        again_img = font.render(again_text, True, blue)
        pygame.draw.rect(screen, green, again_rect)
        screen.blit(again_img, (screen_width // 2 - 80, screen_height // 2 + 10))


#main loop
run = True
while run:
        #draw board and markers first
        draw_board()
        draw_markers()

        #handle events
        for event in pygame.event.get():
                #handle game exit
                if event.type == pygame.QUIT:
                        run = False
                #run new game
                if game_over == False:
                        #check for mouseclick
                        if event.type == pygame.MOUSEBUTTONDOWN and clicked == False:
                                clicked = True
                        if event.type == pygame.MOUSEBUTTONUP and clicked == True:
                                clicked = False
                                pos = pygame.mouse.get_pos()
                                cell_x = pos[0] // 100
                                cell_y = pos[1] // 100
                                if markers[cell_y][cell_x] == 0:
                                        markers[cell_y][cell_x] = player
                                        player *= -1
                                        check_game_over()
                        
        #check if game has been won
        if game_over == True:
                draw_game_over(winner)
                #check for mouseclick to see if we clicked on Play Again
                if event.type == pygame.MOUSEBUTTONDOWN and clicked == False:
                        clicked = True
                if event.type == pygame.MOUSEBUTTONUP and clicked == True:
                        clicked = False
                        pos = pygame.mouse.get_pos()
                        if again_rect.collidepoint(pos):
                                #reset variables
                                game_over = False
                                player = 1
                                pos = (0, 0)
                                markers = []
                                winner = 0
                                #create empty board_layout x board_layout to represent the grid
                                markers = [[0 for i in range(board_layout)] for j in range(board_layout)]

        #update display
        pygame.display.update()
pygame.quit()



























