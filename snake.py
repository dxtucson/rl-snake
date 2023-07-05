import tkinter
from enum import Enum
from random import randint

root = tkinter.Tk()
WIDTH = 440
HEIGHT = 440
SNAKE_W = 20
UPDATE_INT = 100

snake_list = []
generation = 0
generation_str = tkinter.StringVar()
generation_str.set('Generation: 1')
score = 0
score_str = tkinter.StringVar()
score_str.set('Score: 0')
reward_index = 0
pending_action = False

root.geometry(f'{WIDTH}x{HEIGHT + 40}')
canvas = tkinter.Canvas(root, bg='black', height=HEIGHT + 40, width=WIDTH)

# snake area
for rect in range(400):
    rect_x = SNAKE_W + (rect % SNAKE_W) * SNAKE_W
    rect_y = SNAKE_W + (int(rect / 20)) * SNAKE_W
    canvas.create_rectangle(rect_x, rect_y, rect_x + SNAKE_W, rect_y + SNAKE_W, fill='black', outline='', tags=f'{rect}')

# border
canvas.create_rectangle(0, 0, WIDTH, SNAKE_W, fill='red', outline='')
canvas.create_rectangle(WIDTH - SNAKE_W, 0, WIDTH, HEIGHT, fill='red', outline='')
canvas.create_rectangle(0, HEIGHT - SNAKE_W, WIDTH, HEIGHT, fill='red', outline='')
canvas.create_rectangle(0, 0, SNAKE_W, HEIGHT, fill='red', outline='')

generation_label = tkinter.Label(root, font='Arial 12 bold', fg='white', bg='black', textvariable=generation_str)
generation_label.place(x=SNAKE_W, y=HEIGHT + 20, anchor='w')
score_label = tkinter.Label(root, font='Arial 12 bold', fg='white', bg='black', textvariable=score_str)
score_label.place(x=WIDTH - SNAKE_W, y=HEIGHT + 20, anchor='e')

# add to window and show
canvas.pack()


class Status(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    DIED = 100


status = Status.DIED


def check_wall(head):
    return


def update_reward_for_new_head(head):
    if reward_index > 0:
        global score
        if head == reward_index:
            score = score + 1
            score_str.set(f'Score: {score}')
            create_reward()
            return
    index = snake_list.pop()
    all_items = canvas.find_withtag(f'{index}')
    if all_items:
        canvas.itemconfig(all_items[0], fill='black', outline='')


def draw_snake():
    for index in snake_list:
        all_items = canvas.find_withtag(f'{index}')
        if all_items:
            canvas.itemconfig(all_items[0], fill='white', outline='green', width='2')


def create_reward():
    global reward_index
    while reward_index == 0 or reward_index in snake_list:
        reward_index = randint(1, 400)
    items = canvas.find_withtag(f'{reward_index}')
    if items:
        canvas.itemconfig(items[0], fill='yellow')


def snake_up():
    global pending_action, status
    if snake_list:
        new_head = snake_list[0] - 20
        if new_head <= 0:
            status = Status.DIED
            return
        snake_list.insert(0, new_head)
        update_reward_for_new_head(new_head)
        draw_snake()
        pending_action = False


def snake_right():
    global pending_action, status
    if snake_list:
        new_head = snake_list[0] + 1
        if new_head % 20 == 1:
            status = Status.DIED
            return
        snake_list.insert(0, new_head)
        update_reward_for_new_head(new_head)
        draw_snake()
        pending_action = False


def snake_down():
    global pending_action, status
    if snake_list:
        new_head = snake_list[0] + 20
        if new_head > 400:
            status = Status.DIED
            return
        snake_list.insert(0, new_head)
        update_reward_for_new_head(new_head)
        draw_snake()
        pending_action = False


def snake_left():
    global pending_action, status
    if snake_list:
        new_head = snake_list[0] - 1
        if new_head % 20 == 0:
            status = Status.DIED
            return
        snake_list.insert(0, new_head)
        update_reward_for_new_head(new_head)
        draw_snake()
        pending_action = False


def reset(self):
    global snake_list, status, score
    for index in snake_list:
        items = canvas.find_withtag(f'{index}')
        if items:
            canvas.itemconfig(items[0], fill='black', outline='')
    snake_list = [184, 183, 182]
    draw_snake()
    score = 0
    score_str.set(f'Score: 0')
    status = Status.RIGHT
    update()


def update():
    if status == Status.UP:
        snake_up()
    elif status == Status.RIGHT:
        snake_right()
    elif status == Status.DOWN:
        snake_down()
    elif status == Status.LEFT:
        snake_left()
    if status != Status.DIED:
        root.after(UPDATE_INT, update)


def arrow_up(self):
    global status, pending_action
    if not pending_action and status != Status.DIED and status != Status.DOWN:
        status = Status.UP
        pending_action = True


def arrow_right(self):
    global status, pending_action
    if not pending_action and status != Status.DIED and status != Status.LEFT:
        status = Status.RIGHT
        pending_action = True


def arrow_down(self):
    global status, pending_action
    if not pending_action and status != Status.DIED and status != Status.UP:
        status = Status.DOWN
        pending_action = True


def arrow_left(self):
    global status, pending_action
    if not pending_action and status != Status.DIED and status != Status.RIGHT:
        status = Status.LEFT
        pending_action = True


root.bind('<Up>', arrow_up)
root.bind('<Right>', arrow_right)
root.bind('<Down>', arrow_down)
root.bind('<Left>', arrow_left)
root.bind('<space>', reset)
reset(None)
create_reward()
root.mainloop()
