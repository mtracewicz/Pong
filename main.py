import os
import pygame
from game.game_elements.board import Board
from game.constants.constants import Constants
from ml.data_gatherer import DataGatherer
from ml.neural_network import NeuralNetwork

def main():
    run = True
    clock = pygame.time.Clock()
    pygame.init()
    pygame.display.set_caption('Pong')
    screen = pygame.display.set_mode([Constants.WIDTH, Constants.HEIGHT])

    nn = NeuralNetwork.load_model() if os.path.exists('model.txt') else NeuralNetwork([5,4])
    dg = DataGatherer([5])

    board = Board(screen)
    player_direction = 0

    while run:
        clock.tick(Constants.FPS)
        board.move_ball()
        board.detect_hit()
        board.draw()
        pygame.display.update()

        current_data = board.get_data()

        if board.is_on_learning_side():
            dg.record(current_data)
        else:
            nn.fit(dg.get_data()[:,:3],dg.get_data()[:,4])

        ai_direction = nn.predict(current_data)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    player_direction = -1
                if event.key == pygame.K_DOWN:
                    player_direction = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN or event.key == pygame.K_UP:
                    player_direction = 0

        board.move_paddle('p1', player_direction)
        board.move_paddle('p2', ai_direction)

    nn.save_model()
    pygame.quit()


if __name__ == "__main__":
    main()
