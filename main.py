# Import ------------------------------------------------------------------------------------------------------------

import pygame
import sys





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
        pass

    elif convert_btn.draw():
        pass

    elif save_btn.draw():
        pass

    elif reset_btn.draw():
        pass

    # draw grid line
    if draw_rect:
        drawGrid()

    # refresh screen
    pygame.display.flip()