import numpy as np
import pygame
import random
import copy

pygame.init()

x_velocity = 50  # Speed at which the pipes move to the side
gen_pop = 50  # Population size of each generation
percentage = 0.1  # Percentage of population from each generation chosen to reproduce


# Main game class
class Game:
    def __init__(self):
        self.font = pygame.font.SysFont('freesansbold.ttf', 30)
        self.current_gen = 1
        self.screen = pygame.display.set_mode([288, 512])
        self.background = pygame.image.load('assets/background-day.png')
        self.bird_group = pygame.sprite.Group()

        for i in range(gen_pop):
            self.bird_group.add(Bird([np.random.rand(4, 5), np.random.rand(4),
                                      np.random.rand(2, 4), np.random.rand(2)]))

        self.base_group = pygame.sprite.Group()
        self.base_group.add(Obstacale("Base", 0))
        self.base_group.add(Obstacale("Base", 336))

        self.reset_game()

    def reset_game(self):
        self.pipe_group = pygame.sprite.Group()
        self.create_pipe(336)
        self.create_pipe(536)

        self.winners = []

    def update(self, dt, game_speed):
        # Pipes and bases are in sperate groups because they have to be drawn in a specific order
        self.pipe_group.update(dt)
        self.base_group.update(dt)

        # Kills birds
        for entity in self.bird_group:
            if pygame.sprite.spritecollideany(entity, self.pipe_group) or entity.rect.y < 0 or entity.rect.y > 376:
                entity.kill()
                if len(self.bird_group.sprites()) < (gen_pop * percentage):
                    self.winners.append(entity)

        self.bird_group.update(dt, self.pipe_group)

        pipes = self.pipe_group.sprites()

        if len(pipes) == 2:
            self.create_pipe(pipes[0].rect.x + 200)

        # Draws everything
        self.screen.blit(self.background, (0, 0))
        self.pipe_group.draw(self.screen)
        self.base_group.draw(self.screen)
        self.bird_group.draw(self.screen)

        text = self.font.render("Gen:" + str(self.current_gen), True, (0, 0, 0))
        self.screen.blit(text, (0, 512 - text.get_size()[1]))

        text = self.font.render("Speed x" + str(game_speed) + " (+/-)", True, (0, 0, 0))
        self.screen.blit(text, (288 - text.get_size()[0], 512 - text.get_size()[1]))

        if len(self.bird_group.sprites()) == 0:
            self.end_gen()

    # Takes the winners from the previous generation to create the next generation
    def end_gen(self):
        # The parents for a new bird a selected based on weighted probabilities based on how far they have traveled
        # Determines said probabilities
        total = 0
        P = []
        for w in self.winners:
            total += w.distance_traveled
            P.append(w.distance_traveled)

        P = np.array(P) / total

        for n in range(int(gen_pop - (gen_pop * percentage))):
            parents = np.random.choice(self.winners, 2, replace=False, p=P)
            weights = copy.deepcopy(parents[0].weights)

            for i in range(len(weights)):
                shape = np.shape(weights[i])
                for j in range(shape[0]):
                    # Splicing
                    p = random.random()
                    if random.random() < p:
                        weights[i][j] = parents[1].weights[i][j]

                    # Mutation
                    mutation = (2 * np.random.rand(*shape)) - 1

                    if len(shape) == 1:
                        indices = np.random.choice(np.arange(mutation.size), replace=False,
                                                   size=int(mutation.size * 0.95))
                    else:
                        indices = np.random.choice(shape[1] * shape[0], replace=False,
                                                   size=int(shape[1] * shape[0] * 0.95))

                    mutation[np.unravel_index(indices, shape)] = 0
                    weights[i] += mutation

            self.bird_group.add(Bird(weights))

        # The winners are also part of the new generation in case the mutated offspring turn out to be worse
        for winner in self.winners:
            self.bird_group.add(Bird(winner.weights))

        self.current_gen += 1
        self.reset_game()

    def create_pipe(self, x):
        y = np.random.randint(250) + 80
        self.pipe_group.add(Obstacale("pipe_top", x, y - 375))
        self.pipe_group.add(Obstacale("pipe_bottom", x, y + 55))


class Bird(pygame.sprite.Sprite):
    def __init__(self, weights):
        super(Bird, self).__init__()

        image = pygame.image.load('assets/yellowbird.png')
        self.image = pygame.Surface(image.get_size(), pygame.SRCALPHA)
        self.image.blit(image, (0, 0))
        self.rect = self.image.get_rect()
        self.distance_traveled = 0
        self.weights = weights  # Weights contains a list consisting of the NN weights and biases

        self.y = 188

        self.rect.x = 100
        self.rect.y = self.y

        self.y_velocity = 0

    def update(self, *args, **kwargs):
        dt = args[0]
        pipes = args[1].sprites()

        inputs = [self.rect.y, self.y_velocity]  # Inputs for the neural network

        # Finds the closest pipes and finds the inputs
        for i in range(len(pipes)):
            if (pipes[i].rect.x + 52) > self.rect.x:
                inputs.append(pipes[i].rect.y)
                break

        inputs.append(pipes[i + 1].rect.y)
        inputs.append(pipes[i + 1].rect.x - self.rect.x)

        self.y_velocity -= (10 * dt)  # Gravity

        outputs = self.neural_network(np.array(inputs))

        # Jump
        if outputs[0] > outputs[1]:
            self.y_velocity = 5

        self.y -= self.y_velocity
        self.distance_traveled += (x_velocity * dt)

        self.rect.y = self.y

    def neural_network(self, inputs):
        k = inputs
        for i in range((len(self.weights) // 2) - 1):
            k = np.maximum(0, np.dot(self.weights[2 * i], k) + self.weights[2 * i + 1])
        return np.dot(self.weights[-2], k) + self.weights[-1]


# Class for the base and the pipes
class Obstacale(pygame.sprite.Sprite):
    def __init__(self, type, x, y=0):
        super(Obstacale, self).__init__()
        self.type = type

        if type == "pipe_top":
            image = pygame.image.load('assets/pipe.png')
            image = pygame.transform.rotate(image, 180)

        elif type == "pipe_bottom":
            image = pygame.image.load('assets/pipe.png')

        else:
            image = pygame.image.load('assets/base.png')
            y = 400

        self.image = pygame.Surface(image.get_size(), pygame.SRCALPHA)
        self.image.blit(image, (0, 0))
        self.rect = self.image.get_rect()

        self.rect.x = x
        self.rect.y = y

        self.x = x

    def update(self, *args, **kwargs):
        dt = args[0]

        # Moves the object or kills it if its a pipe that has moved offscreen
        if self.rect.x < -self.image.get_size()[0]:
            if self.type == "Base":
                self.x += 650
            else:
                self.kill()
        else:
            self.x -= (x_velocity * dt)

        self.rect.x = self.x


clock = pygame.time.Clock()
game = Game()
running = True
game_speed = 1

# Main game loop
while running:

    # User input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_PLUS and game_speed < 16:
                game_speed *= 2

            elif event.key == pygame.K_MINUS and game_speed > 1:
                game_speed = game_speed // 2

    # Update and draw
    dt = clock.tick(30 * np.sqrt(game_speed)) / 1000
    game.update(dt * game_speed, game_speed)
    pygame.display.flip()

pygame.quit()
