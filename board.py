import pygame
import random
from color import Color
from constants import Constants


class Board():
    def __init__(self, screen):
        self._screen = screen
        self._paddles = {
            'p1': Constants.HEIGHT//2 - 50,
            'p2': Constants.HEIGHT//2-50
        }
        self._ball = {
            'x': Constants.WIDTH//2,
            'y': Constants.HEIGHT//2,
            'vx': 5 if random.randint(1, 3) == 1 else -5,
            'vy': 5
        }
        self._score = {
            'p1': 0,
            'p2': 0
        }

    def draw(self):
        self._draw_board()
        self._draw_paddles()
        self._draw_ball()
        self._draw_score()

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

    def _draw_ball(self):
        pygame.draw.circle(
            self._screen,
            Color.WHITE,
            (
                self._ball['x'],
                self._ball['y']
            ),
            10)

    def move_ball(self):
        self._ball['x'] += self._ball['vx']
        self._ball['y'] += self._ball['vy']

        if self._is_ball_vertically_out_of_bounds():
            self._ball['vy'] = -self._ball['vy']

        if self._is_ball_horizontally_out_of_bounds():
            if self._ball['x'] <= 0:
                self._score['p2'] += 1
            if self._ball['x'] >= Constants.WIDTH:
                self._score['p1'] += 1
            self._reset_ball()

    def _is_ball_vertically_out_of_bounds(self):
        return self._ball['y'] <= 0 or self._ball['y'] >= Constants.HEIGHT

    def _is_ball_horizontally_out_of_bounds(self):
        return self._ball['x'] <= 0 or self._ball['x'] >= Constants.WIDTH

    def _reset_ball(self):
        self._ball['x'] = Constants.WIDTH//2
        self._ball['y'] = Constants.HEIGHT//2
        self._ball['vx'] = 5 if random.randint(1, 3) == 1 else - 5
        self._ball['vy'] = 5

    def detect_hit(self):
        if self._detect_hit_p1():
            self._ball['vx'] = -self._ball['vx']
            self._ball['vy'] = 0.05 * (
                self._ball['y'] -
                self._paddles['p1'] +
                Constants.PADDLE_HEIGHT / 2
            )

        if self._detect_hit_p2():
            self._ball['vx'] = -self._ball['vx']
            self._ball['vy'] = 0.05 * (
                self._ball['y'] -
                self._paddles['p2'] +
                Constants.PADDLE_HEIGHT / 2
            )

    def _detect_hit_p1(self):
        return self._ball['x']-5 <= 25 and self._ball['x']-5 >= 20 and self._ball['y'] >= self._paddles['p1'] and self._ball['y'] <= self._paddles['p1']+Constants.PADDLE_HEIGHT

    def _detect_hit_p2(self):
        return self._ball['x']+5 >= Constants.WIDTH-25 and self._ball['x']+5 >= Constants.WIDTH-15 and self._ball['y'] >= self._paddles['p2'] and self._ball['y'] <= self._paddles['p2']+Constants.PADDLE_HEIGHT

    def _draw_score(self):
        font = pygame.font.SysFont(None, 50)
        label_p1 = font.render(f"P1 : {self._score['p1']}", 1, Color.WHITE)
        label_p2 = font.render(f"P2 : {self._score['p2']}", 1, Color.WHITE)
        self._screen.blit(label_p1, (50, 0))
        self._screen.blit(
            label_p2,
            (
                Constants.WIDTH-label_p2.get_width()-50,
                0
            )
        )
