# Import ------------------------------------------------------------------------------------------------------------

# ui
import pygame
import sys

# utilities
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk,Image 
import cv2

# dehaze algo
from assets.algo.single_image_haze_removal import remove_haze_utility





# Initialize ---------------------------------------------------------------------------------------------------------

# pygame
pygame.init()

# screen
screen_width, screen_height = ( 1244, 712)
bg_color = ( 114, 193, 238)
pygame.display.set_caption( "Satlens")
screen = pygame.display.set_mode( ( screen_width, screen_height))

# misc
tile_size = 25
draw_rect = False
filename = ''





# Classes ------------------------------------------------------------------------------------------------------------
class Button():
    def __init__( self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.clicked = False

    def draw( self):
        action = False
        pos = pygame.mouse.get_pos()
        if self.rect.collidepoint( pos):
            if pygame.mouse.get_pressed()[0] and not self.clicked:
                action = True
                self.clicked = True
        if not pygame.mouse.get_pressed()[0]:
            self.clicked = False
        screen.blit( self.image, self.rect)
        if draw_rect:
            pygame.draw.rect( screen, ( 255, 255, 255), self.rect, 4)

        return action





# Fuctions ------------------------------------------------------------------------------------------------------------

# btn controls
def open_btn_handler():
    global placeholder_img_1, filename
    filename = filedialog.askopenfilename()
    placeholder_img_1 = pygame.image.load(filename)
    pygame.display.update()

def convert_btn_handler():
    global placeholder_img_2, filename
    remove_haze_utility(filename)
    placeholder_img_2 = pygame.image.load('result.jpg')
    pygame.display.update()

def save_btn_handler():
    files = [ ('JPEG image', '*.jpg'),
             ('PNG image', '*.png')]
    filename = filedialog.asksaveasfilename(filetypes = files, defaultextension = files)
    dehazed_img = cv2.imread('result.jpg')
    cv2.imwrite(filename, dehazed_img)

def reset_btn_handler():
    global placeholder_img_1, placeholder_img_2, filename
    filename = ''
    placeholder_img_1 = pygame.image.load( 'assets/images/placeholder_image.png')
    placeholder_img_2 = pygame.image.load( 'assets/images/placeholder_image.png')
    pygame.display.update()

# utility functions
def drawGrid():
    for line in range( 0, 60):
        pygame.draw.line( screen, ( 255, 255, 255), ( 0, line * tile_size), ( screen_width, line * tile_size))
        pygame.draw.line( screen, ( 255, 255, 255), ( line * tile_size, 0), ( line * tile_size, screen_width))
    for line in range( 0, 60):
        pygame.draw.line( screen, ( 0, 0, 0), ( 0, line * tile_size * 4), ( screen_width, line * tile_size * 4), 5)
        pygame.draw.line (screen, ( 0, 0, 0), ( line * tile_size * 4, 0), ( line * tile_size * 4, screen_width), 5)






# Load Assets ------------------------------------------------------------------------------------------------------------

# ui img
image_bg = pygame.image.load( 'assets/images/image_bg.png')
logo_img = pygame.image.load( 'assets/images/logo.png')

# btn img
open_btn_img    = pygame.image.load( 'assets/btn/open.png')
convert_btn_img = pygame.image.load( 'assets/btn/convert.png')
save_btn_img    = pygame.image.load( 'assets/btn/save.png')
reset_btn_img   = pygame.image.load( 'assets/btn/reset.png')

# placeholder image
placeholder_img_1 = pygame.image.load( 'assets/images/placeholder_image.png')
placeholder_img_2 = pygame.image.load( 'assets/images/placeholder_image.png')

# btn
open_btn = Button( 60, 617, open_btn_img)
convert_btn = Button( 160, 617, convert_btn_img)
save_btn = Button( 260, 617, save_btn_img)
reset_btn = Button( 360, 617, reset_btn_img)





# Game Loop ------------------------------------------------------------------------------------------------------------

# loop over
while True:

    # fill bg color
    screen.fill( bg_color)

    # fill images
    screen.blit( image_bg, ( 40, 40))
    screen.blit( image_bg, ( 652, 40))
    screen.blit( placeholder_img_1, ( 60, 60))
    screen.blit( placeholder_img_2, ( 672, 60))
    screen.blit( logo_img, ( 987, 617))

    # check for events
    for event in pygame.event.get():

        # check basic actions
        if event.type == pygame.QUIT:
            sys.exit()

        # check keypress
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                sys.exit()
            if event.key == pygame.K_l:
                draw_rect = not draw_rect

    # check buttons
    if open_btn.draw():
        open_btn_handler()

    if convert_btn.draw():
        convert_btn_handler()

    if save_btn.draw():
        save_btn_handler()

    if reset_btn.draw():
        reset_btn_handler()

    # draw grid line
    if draw_rect:
        drawGrid()

    # refresh screen
    pygame.display.flip()