import pygame
from board import Board
from constants import Constants


def main():
    run = True
    clock = pygame.time.Clock()
    pygame.init()
    pygame.display.set_caption('Pong')
    screen = pygame.display.set_mode([Constants.WIDTH, Constants.HEIGHT])

    board = Board(screen)
    paddle_movement = 0

    while run:
        clock.tick(Constants.FPS)
        board.move_ball()
        board.detect_hit()
        board.draw()
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    paddle_movement = -1
                if event.key == pygame.K_DOWN:
                    paddle_movement = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN or event.key == pygame.K_UP:
                    paddle_movement = 0

        if paddle_movement != 0:
            board.move_paddle(paddle_movement)

    pygame.quit()


if __name__ == "__main__":
    main()
