import os
import numpy as np
import pygame

from game.constants.constants import Constants
from game.constants.modes import Mode
from game.game_elements.board import Board
from game.game_elements.menu import Menu
from ml.data_gatherer import DataGatherer
from ml.neural_network import NeuralNetwork


class Main():
    def __init__(self):
        self.player_direction = 0
        self.ai_direction = 0

        self.clock = pygame.time.Clock()
        pygame.init()
        pygame.display.set_caption('Pong')
        self.screen = pygame.display.set_mode(
            [Constants.WIDTH, Constants.HEIGHT])

        self.nn = None
        self.dg = None

        self.board = Board(self.screen)
        self.menu = Menu(self.screen)
        self.mode = Mode.MENU
        self.game_start = True

    def handle_game(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.player_direction = -1
            if event.key == pygame.K_DOWN:
                self.player_direction = 1
            if event.key == pygame.K_ESCAPE:
                self.mode = Mode.MENU
                self.nn.save_model()
            if event.key == pygame.K_SPACE:
                self.nn.fit(self.dg.get_data())

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_DOWN or event.key == pygame.K_UP:
                self.player_direction = 0

    def handle_menu(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                direction = -1
            elif event.key == pygame.K_DOWN:
                direction = 1
            else:
                direction = 0
            self.menu.change_selected(direction)

            if event.key == pygame.K_SPACE:
                self.handle_menu_click(self.menu.get_selected())

    def handle_menu_click(self, selected):
        if selected == 0:
            self.nn = NeuralNetwork()
            self.dg = DataGatherer()
            self.board.add_dg(self.dg)
            self.mode = Mode.GAME
        elif selected == 1:
            self.nn = NeuralNetwork.load_model() if os.path.exists(
                os.path.join('model', 'weights.npy')) else NeuralNetwork()
            self.dg = DataGatherer(np.load(os.path.join("model", "data.npy"))) if os.path.exists(
                os.path.join("model", "data.npy"))else DataGatherer()
            self.mode = Mode.GAME
            self.board.add_dg(self.dg)
        elif selected == 2:
            self.save()
        else:
            self.end_game()

    def play(self):
        while True:
            self.clock.tick(Constants.FPS)

            if self.mode == Mode.MENU:
                self.menu.draw_menu()
            else:
                self.board.move_ball()
                self.board.detect_hit()
                self.board.draw()

                current_data = self.board.get_data()

                self.dg.record(current_data)

                if current_data[2] > 0:
                    self.ai_direction = self.nn.predict(
                        current_data[:-1])[-1][0]
                    self.ai_direction = 1 if self.ai_direction > current_data[-1] else -1

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.end_game()
                if self.mode == Mode.MENU:
                    self.handle_menu(event)
                else:
                    self.handle_game(event)

            if self.mode == Mode.GAME and not self.game_start:
                self.board.move_paddle('p1', self.player_direction)
                self.board.move_paddle('p2', self.ai_direction)

            elif self.mode == Mode.GAME:
                self.game_start = False

    def end_game(self):
        pygame.quit()
        exit(0)

    def save(self):
        if self.nn is not None:
            self.nn.save_model()
        if self.dg is not None:
            self.dg.save_data()


if __name__ == "__main__":
    game = Main()
    game.play()
