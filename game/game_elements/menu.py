import pygame
from game.constants.constants import Constants
from game.constants.color import Color

class Menu():
    def __init__(self, screen):
        self._selected = 0
        self._menu_items_text = ['1. Play','2. Load model','3. Exit']
        self._font = pygame.font.SysFont(None, 50)
        self._screen = screen

    def draw_menu(self):
        menu_start_position = Constants.HEIGHT//2 - 100
        menu_horizontal_position = Constants.WIDTH//2-100
        for i in range(len(self._menu_items_text)):
            item = self._font.render(self._menu_items_text[i], 1, Color.RED if i == self._selected else Color.WHITE)
            self._screen.blit(item, (menu_horizontal_position, menu_start_position+i*50))

    def change_selected(self, direction):
        self._selected += direction
        self._selected %= 3
    
    def get_selected(self):
        return self._selected