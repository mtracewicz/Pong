import pygame
from color import Color
from constants import Constants


class Board():
    def __init__(self, screen):
        self._screen = screen
        self._paddles = {'p1': Constants.HEIGHT//2 -
                         50, 'p2': Constants.HEIGHT//2-50}

    def draw(self):
        self._draw_board()
        self._draw_paddles()

    def _draw_board(self):
        self._screen.fill(Color.BLACK)
        for i in range(0, Constants.HEIGHT, 50):
            pygame.draw.rect(self._screen, Color.WHITE,
                             (Constants.WIDTH//2, i, 10, 25))

    def _draw_paddles(self):
        pygame.draw.rect(
            self._screen,
            Color.WHITE,
            (
                15,
                self._paddles['p1'],
                10,
                Constants.PADDLE_HEIGHT
            ))
        pygame.draw.rect(
            self._screen,
            Color.WHITE,
            (
                Constants.WIDTH-25,
                self._paddles['p2'],
                10,
                Constants.PADDLE_HEIGHT
            ))

    def move_paddle(self, direction):
        new_position = self._paddles['p1'] + direction*Constants.SINGLE_MOVE
        if self._will_paddle_be_visible(new_position):
            self._paddles['p1'] = new_position

    def _will_paddle_be_visible(self, new_position):
        return (new_position >= 0 and new_position <= Constants.HEIGHT-Constants.PADDLE_HEIGHT)
