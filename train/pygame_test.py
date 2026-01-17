import pygame

def main():
    pygame.init()
    screen = pygame.display.set_mode((300, 200))
    pygame.display.set_caption("Wise Snake - pygame test")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((30, 30, 30))
        pygame.display.flip()

    pygame.quit()
    print("pygame OK")

if __name__ == "__main__":
    main()
