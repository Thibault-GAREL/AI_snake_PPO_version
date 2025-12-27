import pygame
import random
from dataclasses import dataclass

import ia

show = False
player = False

info = False

action_dim = 4


stop_iteration = 100

@dataclass
class Snake:
    x: int
    y: int

class Manager_snake:
    def __init__(self):
        self.list_snake = []
        self.lenght = 0
        self.direction = "RIGHT"
        self.moved = True  # Ajout du drapeau moved

    def add_snake(self, added_snake):
        self.list_snake.append(added_snake)
        self.lenght += 1

    def draw_snake(self):
        for index, selected_snake in enumerate(self.list_snake, start=0):
            pygame.draw.rect(display, GREEN, (selected_snake.x, selected_snake.y, rect_width, rect_height))
            # draw_cherckerboard()
            if index == 0:
                if self.direction == "UP":
                    pygame.draw.rect(display, BLACK, (selected_snake.x + 10, selected_snake.y + 5, 5, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 15, selected_snake.y + 5, 5, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, 5, rect_height))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 5, selected_snake.y, 5, rect_height))
                if self.direction == "DOWN":
                    pygame.draw.rect(display, BLACK, (selected_snake.x + 10, selected_snake.y + rect_width - 10, 5, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 15, selected_snake.y + rect_width - 10, 5, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, 5, rect_height))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 5, selected_snake.y, 5, rect_height))
                if self.direction == "RIGHT":
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_height - 10, selected_snake.y + 10, 5, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_height - 10, selected_snake.y + rect_height - 15, 5, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, rect_width, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y + rect_width - 5, rect_height, 5))
                if self.direction == "LEFT":
                    pygame.draw.rect(display, BLACK, (selected_snake.x + 5, selected_snake.y + 10, 5, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + 5, selected_snake.y + rect_height - 15, 5, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, rect_width, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y + rect_width - 5, rect_height, 5))

            if index == self.lenght-1:
                if self.list_snake[index].y != self.list_snake[index-1].y:
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, 5, rect_height))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 5, selected_snake.y, 5, rect_height))
                if self.list_snake[index].x != self.list_snake[index-1].x:
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, rect_width, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y + rect_width - 5, rect_height, 5))

            if index < self.lenght-1 and index !=0:
                if (self.list_snake[index-1].x<selected_snake.x and selected_snake.x<self.list_snake[index+1].x) or (self.list_snake[index+1].x<selected_snake.x and selected_snake.x<self.list_snake[index-1].x):
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, rect_width, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y + rect_height - 5, rect_width, 5))
                if (self.list_snake[index-1].y<selected_snake.y and selected_snake.y<self.list_snake[index+1].y) or (self.list_snake[index+1].y<selected_snake.y and selected_snake.y<self.list_snake[index-1].y):
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, 5, rect_height))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_height - 5, selected_snake.y, 5, rect_width))
                if (self.list_snake[index].y < self.list_snake[index+1].y and self.list_snake[index].x < self.list_snake[index-1].x) or (self.list_snake[index].y < self.list_snake[index-1].y and self.list_snake[index].x < self.list_snake[index+1].x):
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, 5, rect_height))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, rect_width, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 5, selected_snake.y + rect_height - 5, 5, 5))
                if (self.list_snake[index].y > self.list_snake[index+1].y and self.list_snake[index].x < self.list_snake[index-1].x) or (self.list_snake[index].y > self.list_snake[index-1].y and self.list_snake[index].x < self.list_snake[index+1].x):
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, 5, rect_height))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y + rect_height - 5, rect_width, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 5, selected_snake.y, 5, 5))
                if (self.list_snake[index].y > self.list_snake[index+1].y and self.list_snake[index].x > self.list_snake[index-1].x) or (self.list_snake[index].y > self.list_snake[index-1].y and self.list_snake[index].x > self.list_snake[index+1].x):
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y + rect_width - 5, rect_width, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 5, selected_snake.y, 5, rect_height))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, 5, 5))
                if (self.list_snake[index].y < self.list_snake[index+1].y and self.list_snake[index].x > self.list_snake[index-1].x) or (self.list_snake[index].y < self.list_snake[index-1].y and self.list_snake[index].x > self.list_snake[index+1].x):
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, rect_width, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 5, selected_snake.y, 5, rect_height))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y + rect_height - 5, 5, 5))

    def move(self):
        head_x = self.list_snake[0].x
        head_y = self.list_snake[0].y

        if self.direction == "UP":
            head_y -= 50
        if self.direction == "DOWN":
            head_y += 50
        if self.direction == "RIGHT":
            head_x += 50
        if self.direction == "LEFT":
            head_x -= 50

        new_head = Snake(head_x, head_y)

        if not ((0 <= new_head.x < width) and (0 <= new_head.y < height)):
            return False

        if new_head in self.list_snake:
            return False

        self.list_snake.insert(0, new_head)
        self.list_snake.pop(-1)
        self.moved = True
        return True

    def print_snake(self):
        for index, selected_snake in enumerate(self.list_snake, start=0):
            print(f"snake n°{index} = {selected_snake}")

@dataclass
class food:
    x: int
    y: int

def generated_food (my_snake):
    list_position = []
    for x in range (0, int(width/rect_width)):
        for y in range(0, int(height/rect_height)):
            list_position.append((x * rect_width, y * rect_height))

    if not list_position:
        print("C'est gagné!")
    for co_snake in my_snake.list_snake:
        if (co_snake.x, co_snake.y) in list_position:
            list_position.remove((co_snake.x, co_snake.y))
    random_food = random.choice(list_position)
    return food(random_food[0], random_food[1])


def distance_bord_west (my_snake):
    distance_mini = my_snake.list_snake[0].x
    for snake_selected in my_snake.list_snake:
        if snake_selected.y == my_snake.list_snake[0].y and snake_selected.x < my_snake.list_snake[0].x:
            distance_temp = my_snake.list_snake[0].x - snake_selected.x
            if distance_temp < distance_mini:
                distance_mini = distance_temp
    return distance_mini

def distance_bord_est (my_snake):
    distance_mini = width - 50 - my_snake.list_snake[0].x
    for snake_selected in my_snake.list_snake:
        if snake_selected.y == my_snake.list_snake[0].y and snake_selected.x > my_snake.list_snake[0].x:
            distance_temp = snake_selected.x - my_snake.list_snake[0].x
            if distance_temp < distance_mini:
                distance_mini = distance_temp
    return distance_mini

def distance_bord_north (my_snake):
    distance_mini = my_snake.list_snake[0].y
    for snake_selected in my_snake.list_snake:
        if snake_selected.x == my_snake.list_snake[0].x and snake_selected.y < my_snake.list_snake[0].y:
            distance_temp = my_snake.list_snake[0].y - snake_selected.y
            if distance_temp < distance_mini:
                distance_mini = distance_temp
    return distance_mini

def distance_bord_south (my_snake):
    distance_mini = height - 50 - my_snake.list_snake[0].y
    for snake_selected in my_snake.list_snake:
        if snake_selected.x == my_snake.list_snake[0].x and snake_selected.y > my_snake.list_snake[0].y:
            distance_temp = snake_selected.y - my_snake.list_snake[0].y
            if distance_temp < distance_mini:
                distance_mini = distance_temp
    return distance_mini

def distance_bord_north_est (my_snake):
    distance_mini = ((my_snake.list_snake[0].y)**2 + (my_snake.list_snake[0].y)**2)**0.5 #distance north
    if distance_mini > ((width - 50 - my_snake.list_snake[0].x)**2 + (width - 50 - my_snake.list_snake[0].x)**2)**0.5:
        distance_mini = ((width - 50 - my_snake.list_snake[0].x)**2 + (width - 50 - my_snake.list_snake[0].x)**2)**0.5 #distance est
    for snake_selected in my_snake.list_snake:
        if my_snake.list_snake[0].x != snake_selected.x and my_snake.list_snake[0].y != snake_selected.y:
            pente = (my_snake.list_snake[0].x - snake_selected.x) / (my_snake.list_snake[0].y - snake_selected.y)
            if pente == -1 and my_snake.list_snake[0].x < snake_selected.x:
                distance = ((snake_selected.x - my_snake.list_snake[0].x)**2 + (snake_selected.y - my_snake.list_snake[0].y)**2)**0.5
                if distance < distance_mini:
                    distance_mini = distance
    return distance_mini

def distance_bord_south_est (my_snake):
    distance_mini = ((height - 50 - my_snake.list_snake[0].y)**2 + (height - 50 - my_snake.list_snake[0].y)**2)**0.5 #distance south
    if distance_mini > ((width - 50 - my_snake.list_snake[0].x)**2 + (width - 50 - my_snake.list_snake[0].x)**2)**0.5:
        distance_mini = ((width - 50 - my_snake.list_snake[0].x)**2 + (width - 50 - my_snake.list_snake[0].x)**2)**0.5 #distance est
    for snake_selected in my_snake.list_snake:
        if my_snake.list_snake[0].x != snake_selected.x and my_snake.list_snake[0].y != snake_selected.y:
            pente = (my_snake.list_snake[0].x - snake_selected.x) / (my_snake.list_snake[0].y - snake_selected.y)
            if pente == 1 and my_snake.list_snake[0].x < snake_selected.x:
                distance = ((snake_selected.x - my_snake.list_snake[0].x)**2 + (snake_selected.y - my_snake.list_snake[0].y)**2)**0.5
                if distance < distance_mini:
                    distance_mini = distance
    return distance_mini

def distance_bord_south_west (my_snake):
    distance_mini = ((height - 50 - my_snake.list_snake[0].y)**2 + (height - 50 - my_snake.list_snake[0].y)**2)**0.5 #distance south
    if distance_mini > ((my_snake.list_snake[0].x)**2 + (my_snake.list_snake[0].x)**2)**0.5:
        distance_mini = ((my_snake.list_snake[0].x)**2 + (my_snake.list_snake[0].x)**2)**0.5 #distance west
    for snake_selected in my_snake.list_snake:
        if my_snake.list_snake[0].x != snake_selected.x and my_snake.list_snake[0].y != snake_selected.y:
            pente = (my_snake.list_snake[0].x - snake_selected.x) / (my_snake.list_snake[0].y - snake_selected.y)
            if pente == -1 and snake_selected.x < my_snake.list_snake[0].x:
                distance = ((snake_selected.x - my_snake.list_snake[0].x)**2 + (snake_selected.y - my_snake.list_snake[0].y)**2)**0.5
                if distance < distance_mini:
                    distance_mini = distance
    return distance_mini

def distance_bord_north_west (my_snake):
    distance_mini = ((my_snake.list_snake[0].y)**2 + (my_snake.list_snake[0].y)**2)**0.5 #distance north
    if distance_mini > ((my_snake.list_snake[0].x)**2 + (my_snake.list_snake[0].x)**2)**0.5:
        distance_mini = ((my_snake.list_snake[0].x)**2 + (my_snake.list_snake[0].x)**2)**0.5 #distance west
    for snake_selected in my_snake.list_snake:
        if my_snake.list_snake[0].x != snake_selected.x and my_snake.list_snake[0].y != snake_selected.y:
            pente = (my_snake.list_snake[0].x - snake_selected.x) / (my_snake.list_snake[0].y - snake_selected.y)
            if pente == 1 and snake_selected.x < my_snake.list_snake[0].x:
                distance = ((snake_selected.x - my_snake.list_snake[0].x)**2 + (snake_selected.y - my_snake.list_snake[0].y)**2)**0.5
                if distance < distance_mini:
                    distance_mini = distance
    return distance_mini

def distance_food_north (my_snake, food):
    distance = 0
    if food.y < my_snake.list_snake[0].y and food.x == my_snake.list_snake[0].x:
        distance = my_snake.list_snake[0].y - food.y
    return distance

def distance_food_est (my_snake, food):
    distance = 0
    if my_snake.list_snake[0].x < food.x and food.y == my_snake.list_snake[0].y:
        distance = food.x - my_snake.list_snake[0].x
    return distance

def distance_food_south (my_snake, food):
    distance = 0
    if my_snake.list_snake[0].y < food.y and food.x == my_snake.list_snake[0].x:
        distance = food.y - my_snake.list_snake[0].y
    return distance

def distance_food_west (my_snake, food):
    distance = 0
    if food.x < my_snake.list_snake[0].x and food.y == my_snake.list_snake[0].y:
        distance = my_snake.list_snake[0].x - food.x
    return distance

def distance_food_north_est (my_snake, food):
    distance = 0
    if my_snake.list_snake[0].x != food.x and my_snake.list_snake[0].y != food.y:
        pente = (my_snake.list_snake[0].x - food.x) / (my_snake.list_snake[0].y - food.y)
        if pente == -1 and my_snake.list_snake[0].x < food.x:
            distance = ((food.x - my_snake.list_snake[0].x)**2 + (food.y - my_snake.list_snake[0].y)**2)**0.5
    return distance

def distance_food_south_est (my_snake, food):
    distance = 0
    if my_snake.list_snake[0].x != food.x and my_snake.list_snake[0].y != food.y:
        pente = (my_snake.list_snake[0].x - food.x) / (my_snake.list_snake[0].y - food.y)
        if pente == 1 and my_snake.list_snake[0].x < food.x:
            distance = ((food.x - my_snake.list_snake[0].x) ** 2 + (food.y - my_snake.list_snake[0].y) ** 2) ** 0.5
    return distance

def distance_food_south_west (my_snake, food):
    distance = 0
    if my_snake.list_snake[0].x != food.x and my_snake.list_snake[0].y != food.y:
        pente = (my_snake.list_snake[0].x - food.x) / (my_snake.list_snake[0].y - food.y)
        if pente == -1 and food.x < my_snake.list_snake[0].x:
            distance = ((my_snake.list_snake[0].x - food.x)**2 + (my_snake.list_snake[0].y - food.y)**2)**0.5
    return distance

def distance_food_north_west (my_snake, food):
    distance = 0
    if my_snake.list_snake[0].x != food.x and my_snake.list_snake[0].y != food.y:
        pente = (my_snake.list_snake[0].x - food.x) / (my_snake.list_snake[0].y - food.y)
        if pente == 1 and food.x < my_snake.list_snake[0].x:
            distance = ((my_snake.list_snake[0].x - food.x) ** 2 + (my_snake.list_snake[0].y - food.y) ** 2) ** 0.5
    return distance

def print_display(msg, color, position):
    txt = fonttype.render(msg, True, color)
    rect_text = txt.get_rect(**position)
    display.blit(txt, rect_text)

def draw_cherckerboard ():
    for x in range (0, int(width/rect_width)):
        for y in range(0, int(height/rect_height)):
            pygame.draw.line(display, BLACK_BRIGHTER, (x * rect_width, y * rect_height), (x * rect_width + rect_width, y * rect_height), 2)
            pygame.draw.line(display, BLACK_BRIGHTER, (x * rect_width, y * rect_height), (x * rect_width, y * rect_height + rect_height), 2)






def game_loop(rect_width, rect_height, display, agent):
    iteration = 1
    action = 1
    log_prob = None
    state_value = None

    # food_actuel = food(9 * 50, 5 * 50)

    score = 0

    my_snake = Manager_snake()
    first_snake = Snake(5 * 50, 5 * 50) # ancienement : 5 * 50, 5 * 50
    my_snake.add_snake(first_snake)


    food_actuel = generated_food(my_snake)

    done = False

    while not done and iteration <= stop_iteration:
        if player:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                    if my_snake.moved:
                        if event.key == pygame.K_UP and my_snake.direction != "DOWN":
                            my_snake.direction = "UP"
                            action = 0
                            my_snake.moved = False  # Réinitialise moved
                        if event.key == pygame.K_DOWN and my_snake.direction != "UP":
                            my_snake.direction = "DOWN"
                            action = 2
                            my_snake.moved = False
                        if event.key == pygame.K_RIGHT and my_snake.direction != "LEFT":
                            my_snake.direction = "RIGHT"
                            action = 1
                            my_snake.moved = False
                        if event.key == pygame.K_LEFT and my_snake.direction != "RIGHT":
                            my_snake.direction = "LEFT"
                            action = 3
                            my_snake.moved = False

        if show:
            display.fill(BLACK)

        state = [distance_bord_north(my_snake), distance_bord_north_est(my_snake),
                 distance_bord_est(my_snake), distance_bord_south_est(my_snake),
                 distance_bord_south(my_snake), distance_bord_south_west(my_snake),
                 distance_bord_west(my_snake), distance_bord_north_west(my_snake),

                 distance_food_north(my_snake, food_actuel), distance_food_north_est(my_snake, food_actuel),
                 distance_food_est(my_snake, food_actuel), distance_food_south_est(my_snake, food_actuel),
                 distance_food_south(my_snake, food_actuel), distance_food_south_west(my_snake, food_actuel),
                 distance_food_west(my_snake, food_actuel), distance_food_north_west(my_snake, food_actuel)]

        reward = 0

        # print(state)

        if not player:
            # action = random.randint(0, action_dim)
            # Pour PPO, on a besoin du log_prob et de la state_value
            action, log_prob, state_value = agent.select_action_with_log_prob(state)
            # print(action)

        if action == 0 and my_snake.direction != "DOWN":  # î : 1 / -> : 2 / v : 3 / <- : 4
            my_snake.direction = "UP"
            my_snake.moved = False  # Réinitialise moved
        if action == 2 and my_snake.direction != "UP":
            my_snake.direction = "DOWN"
            my_snake.moved = False
        if action == 1 and my_snake.direction != "LEFT":
            my_snake.direction = "RIGHT"
            my_snake.moved = False
        if action == 3 and my_snake.direction != "RIGHT":
            my_snake.direction = "LEFT"
            my_snake.moved = False

        if my_snake.list_snake[0].x == food_actuel.x and my_snake.list_snake[0].y == food_actuel.y:
            snake_to_add = Snake(my_snake.list_snake[-1].x, my_snake.list_snake[-1].y)
            my_snake.add_snake(snake_to_add)
            food_actuel = generated_food(my_snake)
            score += 1
            reward += 1

        if my_snake.move() == False:
            done = True
            reward -= 1


        if show:
            pygame.draw.rect(display, RED, (food_actuel.x, food_actuel.y, rect_width, rect_height))

            my_snake.draw_snake()
        # my_snake.print_snake()

        if info:
            print("=======================================")


            distance_bord_n = distance_bord_north(my_snake)
            distance_bord_e = distance_bord_est(my_snake)
            distance_bord_s = distance_bord_south(my_snake)
            distance_bord_w = distance_bord_west(my_snake)

            print(f"distance north : {distance_bord_n}")
            print(f"distance est : {distance_bord_e}")
            print(f"distance south : {distance_bord_s}")
            print(f"distance west : {distance_bord_w}")

            print("_______________")

            distance_food_n = distance_food_north(my_snake, food_actuel)
            distance_food_e = distance_food_est(my_snake, food_actuel)
            distance_food_s = distance_food_south(my_snake, food_actuel)
            distance_food_w = distance_food_west(my_snake, food_actuel)

            print(f"distance food north : {distance_food_n}")
            print(f"distance food est : {distance_food_e}")
            print(f"distance food south : {distance_food_s}")
            print(f"distance food west : {distance_food_w}")

            print("_______________")

            distance_food_n_e = distance_food_north_est(my_snake, food_actuel)
            distance_food_s_e = distance_food_south_est(my_snake, food_actuel)
            distance_food_s_w = distance_food_south_west(my_snake, food_actuel)
            distance_food_n_w = distance_food_north_west(my_snake, food_actuel)

            print(f"distance food north est : {distance_food_n_e}")
            print(f"distance food south est : {distance_food_s_e}")
            print(f"distance food south west : {distance_food_s_w}")
            print(f"distance food north west : {distance_food_n_w}")

            print("_______________")

            distance_bord_n_e = distance_bord_north_est(my_snake)
            distance_bord_s_e = distance_bord_south_est(my_snake)
            distance_bord_s_w = distance_bord_south_west(my_snake)
            distance_bord_n_w = distance_bord_north_west(my_snake)

            print(f"distance bord north est : {distance_bord_n_e}")
            print(f"distance bord south est : {distance_bord_s_e}")
            print(f"distance bord south west : {distance_bord_s_w}")
            print(f"distance bord north west : {distance_bord_n_w}")

        # draw_cherckerboard()

        if show:
            print_display(f"Score : {score}", WHITE, {'topleft': (10, 10)})
            # print(f"lenght : {my_snake.lenght}")

        next_state = [distance_bord_north(my_snake), distance_bord_north_est(my_snake),
             distance_bord_est(my_snake), distance_bord_south_est(my_snake),
             distance_bord_south(my_snake), distance_bord_south_west(my_snake),
             distance_bord_west(my_snake), distance_bord_north_west(my_snake),

             distance_food_north(my_snake, food_actuel), distance_food_north_est(my_snake, food_actuel),
             distance_food_est(my_snake, food_actuel), distance_food_south_est(my_snake, food_actuel),
             distance_food_south(my_snake, food_actuel), distance_food_south_west(my_snake, food_actuel),
             distance_food_west(my_snake, food_actuel), distance_food_north_west(my_snake, food_actuel)]

        # Pour PPO, on stocke: state, action, reward, state_value, log_prob, done
        if not player and log_prob is not None:
            agent.buffer.push(state, action, reward, state_value, log_prob, done)

        if show:
            pygame.display.update()
            clock.tick(5)

        # print (iteration)

        iteration += 1

        # pygame.quit()
        # sys.exit()
    return score



width = 800
height = 600

rect_width = 50
rect_height = 50

if show:
    pygame.init()
    display = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Jeu snake")

    clock = pygame.time.Clock()

    fonttype = pygame.font.SysFont(None, 30)

else:
    display = None

BLACK = (40, 40, 60)
BLACK_BRIGHTER = (60, 60, 80)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 244)


