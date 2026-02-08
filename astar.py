import pygame
import math
from queue import PriorityQueue


WIDTH  = 1180
HEIGHT = 780

USE_FILE = input("Load map from file? (y/n): ").strip().lower() == "y"

if USE_FILE:
    MAP_FILE = input("Enter map filename (e.g. map.txt): ").strip()
    ROWS = COLS = None        #from file
    start_x = start_y = None  #from file
    end_x = end_y = None
else:
    ROWS, COLS = map(int, input("Enter grid dimensions (y X x): ").split())
    
    start_y, start_x = map(int, input("Enter position of start node (y X x): ").split())
    while(start_x < 1 or start_x > COLS or start_y < 1 or start_y > ROWS):
        print("Invalid start position. Please enter again.")
        start_y, start_x = map(int, input("Enter position of start node (y X x): ").split())
        
    end_y,   end_x   = map(int, input("Enter position of end node (y X x): ").split())
    while(end_x < 1 or end_x > COLS or end_y < 1 or end_y > ROWS):
        print("Invalid end position. Please enter again.")
        end_y, end_x = map(int, input("Enter position of end node (y X x): ").split())

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Assignment 7")

#COLORS FOR THE DISPLAY
PINK = (255,192,203)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (150, 75, 0)
GREY = (128, 128, 128)
RED = (255, 0, 0)

pygame.font.init()
FONT = pygame.font.SysFont("arial", 8)

BASE_CELL_SIZE = 10
zoom_factor = 1.0
ZOOM_STEP = 0.1
MIN_ZOOM = 0.5
MAX_ZOOM = 3.0

cam_offset_x = 0
cam_offset_y = 0
CAM_SPEED = 20


def logical_to_index(x, y, ROWS, COLS):
    col = x - 1
    row = ROWS - y
    return row, col

def index_to_logical(row, col, ROWS, COLS):
    x = col + 1
    y = ROWS - row
    return x, y


def load_map_from_file(filename):
    """
    Format:
    Map:
    000...
    001...
    with '0','1','s','t'
    """
    with open(filename, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    map_lines = []
    started = False
    for line in lines:
        if not started:
            if line.strip().startswith("0") or "s" in line or "t" in line or "1" in line:
                started = True
                map_lines.append(line)
        else:
            if line.strip() == "":
                break
            map_lines.append(line)

    rows = len(map_lines)
    cols = len(map_lines[0]) if rows > 0 else 0

    start_rc = None
    end_rc = None
    barriers = []

    for r, line in enumerate(map_lines):
        for c, ch in enumerate(line):
            if ch == "1":
                barriers.append((r, c))
            elif ch == "s":
                start_rc = (r, c)
            elif ch == "t":
                end_rc = (r, c)

    return rows, cols, start_rc, end_rc, barriers


class Node:
    def __init__(self, row, col, cell_width, cell_height, total_rows, total_cols):
        self.row = row
        self.col = col
        self.x = col * cell_width
        self.y = row * cell_height

        self.color = WHITE
        self.neighbors = []

        self.cell_width = cell_width
        self.cell_height = cell_height
        self.total_rows = total_rows
        self.total_cols = total_cols

        self.g_value = None
        self.h_value = None
        self.f_value = None

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == GREY

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BROWN

    def is_start(self):
        return self.color == BLUE

    def is_end(self):
        return self.color == RED

    def reset(self):
        self.color = WHITE
        self.f_value = None
        self.g_value = None
        self.h_value = None

    def make_start(self):
        self.color = BLUE

    def make_closed(self):
        self.color = GREY

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BROWN

    def make_end(self):
        self.color = RED

    def make_path(self):
        self.color = PINK

    def draw(self, surf):
        pygame.draw.rect(
            surf,
            self.color,
            (self.x, self.y, self.cell_width, self.cell_height)
        )

        #f func
        if self.f_value is not None:
            f_text = FONT.render(str(self.f_value), True, BLACK)
            f_rect = f_text.get_rect(center=(self.x + self.cell_width // 2, self.y + 6))
            surf.blit(f_text, f_rect)

        #g+h functions
        if self.g_value is not None and self.h_value is not None:
            gh_text = FONT.render(f"{self.g_value}+{self.h_value}", True, BLACK)
            gh_rect = gh_text.get_rect(center=(self.x + self.cell_width // 2, self.y + self.cell_height - 8))
            surf.blit(gh_text, gh_rect)

    def update_neighbors(self, grid):
        self.neighbors = []

        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_cols - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False

#manhattan
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current, draw_func):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw_func()

def a_star_algorithm(draw_func, grid, start, end, use_zoom):
    global zoom_factor, cam_offset_x, cam_offset_y

    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}

    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0

    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    start.g_value = 0
    start.h_value = h(start.get_pos(), end.get_pos())
    start.f_value = f_score[start]

    open_set_hash = {start}

	#speed
    while not open_set.empty():
        pygame.time.delay(1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            if use_zoom and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    zoom_factor = min(MAX_ZOOM, zoom_factor + ZOOM_STEP)
                if event.key == pygame.K_e:
                    zoom_factor = max(MIN_ZOOM, zoom_factor - MIN_ZOOM)

                if event.key == pygame.K_LEFT:
                    cam_offset_x += CAM_SPEED
                if event.key == pygame.K_RIGHT:
                    cam_offset_x -= CAM_SPEED
                if event.key == pygame.K_UP:
                    cam_offset_y += CAM_SPEED
                if event.key == pygame.K_DOWN:
                    cam_offset_y -= CAM_SPEED

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw_func)
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                h_val = h(neighbor.get_pos(), end.get_pos())
                f_val = temp_g_score + h_val
                f_score[neighbor] = f_val

                neighbor.g_value = temp_g_score
                neighbor.h_value = h_val
                neighbor.f_value = f_val

                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        current.g_value = g_score[current]
        current.h_value = h(current.get_pos(), end.get_pos())
        current.f_value = f_score[current]

        draw_func()

        if current != start:
            current.make_closed()

    return False


def save_map_to_txt(grid, rows, cols, filename="map.txt"):
    with open(filename, "w") as f:
        for r in range(rows):
            line_chars = []
            for c in range(cols):
                spot = grid[r][c]
                if spot.is_start():
                    line_chars.append("s")
                elif spot.is_end():
                    line_chars.append("t")
                elif spot.is_barrier():
                    line_chars.append("1")
                else:
                    line_chars.append("0")
            f.write("".join(line_chars) + "\n")


def make_grid_plain(rows, cols, width, height):
    grid = []
    cell_width = width // cols
    cell_height = height // rows
    for i in range(rows):
        grid.append([])
        for j in range(cols):
            spot = Node(i, j, cell_width, cell_height, rows, cols)
            grid[i].append(spot)
    return grid

def draw_grid_plain(win, rows, cols, width, height):
    cell_width = width // cols
    cell_height = height // rows
    for i in range(rows + 1):
        pygame.draw.line(win, GREY, (0, i * cell_height), (width, i * cell_height))
    for j in range(cols + 1):
        pygame.draw.line(win, GREY, (j * cell_width, 0), (j * cell_width, height))

def draw_plain(win, grid, rows, cols, width, height):
    win.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid_plain(win, rows, cols, width, height)
    pygame.display.update()

def get_clicked_pos_plain(pos, rows, cols, width, height):
    x, y = pos
    cell_width = width // cols
    cell_height = height // rows
    col = x // cell_width
    row = y // cell_height
    return row, col


def make_grid_zoom(rows, cols, cell_size):
    grid = []
    for i in range(rows):
        grid.append([])
        for j in range(cols):
            spot = Node(i, j, cell_size, cell_size, rows, cols)
            grid[i].append(spot)
    return grid

def draw_world(surf, grid, rows, cols, cell_size):
    surf.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(surf)
    for i in range(rows + 1):
        pygame.draw.line(surf, GREY, (0, i * cell_size), (cols * cell_size, i * cell_size))
    for j in range(cols + 1):
        pygame.draw.line(surf, GREY, (j * cell_size, 0), (j * cell_size, rows * cell_size))

def render(world_surf, win, zoom_factor, cam_offset_x, cam_offset_y):
    w = int(world_surf.get_width() * zoom_factor)
    h = int(world_surf.get_height() * zoom_factor)
    scaled = pygame.transform.scale(world_surf, (w, h))
    win.fill(WHITE)
    rect = scaled.get_rect()
    rect.x = cam_offset_x
    rect.y = cam_offset_y
    win.blit(scaled, rect)
    pygame.display.update()
    return rect

def get_clicked_pos_zoom(pos, zoom_factor, world_rect, cell_size):
    mx, my = pos
    sx = (mx - world_rect.x) / zoom_factor
    sy = (my - world_rect.y) / zoom_factor
    if sx < 0 or sy < 0:
        return None, None
    col = int(sx // cell_size)
    row = int(sy // cell_size)
    return row, col


def main(win, width, height):
    global ROWS, COLS, start_x, start_y, end_x, end_y
    global zoom_factor, cam_offset_x, cam_offset_y

    if USE_FILE:
        rows, cols, start_rc, end_rc, barriers = load_map_from_file(MAP_FILE)
        ROWS, COLS = rows, cols
        s_row, s_col = start_rc
        e_row, e_col = end_rc

    use_zoom = (ROWS >= 70 or COLS >= 70)

    if not use_zoom:
        grid = make_grid_plain(ROWS, COLS, width, height)

        if USE_FILE:
            start = grid[s_row][s_col]; start.make_start()
            end   = grid[e_row][e_col]; end.make_end()
            for (r, c) in barriers:
                if grid[r][c] not in (start, end):
                    grid[r][c].make_barrier()
        else:
            s_row, s_col = logical_to_index(start_x, start_y, ROWS, COLS)
            e_row, e_col = logical_to_index(end_x, end_y, ROWS, COLS)
            start = grid[s_row][s_col]; start.make_start()
            end   = grid[e_row][e_col]; end.make_end()

        run = True
        while run:
            draw_plain(win, grid, ROWS, COLS, width, height)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

                if pygame.mouse.get_pressed()[0]:
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_pos_plain(pos, ROWS, COLS, width, height)
                    if 0 <= row < ROWS and 0 <= col < COLS:
                        spot = grid[row][col]
                        if not start and spot != end:
                            start = spot; start.make_start()
                        elif not end and spot != start:
                            end = spot; end.make_end()
                        elif spot != end and spot != start:
                            spot.make_barrier()

                elif pygame.mouse.get_pressed()[2]:
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_pos_plain(pos, ROWS, COLS, width, height)
                    if 0 <= row < ROWS and 0 <= col < COLS:
                        spot = grid[row][col]
                        spot.reset()
                        if spot == start: start = None
                        elif spot == end: end = None

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and start and end:
                        for row in grid:
                            for spot in row:
                                spot.update_neighbors(grid)
                        a_star_algorithm(
                            lambda: draw_plain(win, grid, ROWS, COLS, width, height),
                            grid, start, end, use_zoom=False
                        )

                    if event.key == pygame.K_s:
                        save_map_to_txt(grid, ROWS, COLS, "map.txt")
                        print("Map saved to map.txt")

                    if event.key == pygame.K_c:
                        start = None
                        end = None
                        grid = make_grid_plain(ROWS, COLS, width, height)

    else:
        cell_size = BASE_CELL_SIZE
        world_width  = COLS * cell_size
        world_height = ROWS * cell_size
        world_surf = pygame.Surface((world_width, world_height))
        grid = make_grid_zoom(ROWS, COLS, cell_size)

        if USE_FILE:
            start = grid[s_row][s_col]; start.make_start()
            end   = grid[e_row][e_col]; end.make_end()
            for (r, c) in barriers:
                if grid[r][c] not in (start, end):
                    grid[r][c].make_barrier()
        else:
            s_row, s_col = logical_to_index(start_x, start_y, ROWS, COLS)
            e_row, e_col = logical_to_index(end_x, end_y, ROWS, COLS)
            start = grid[s_row][s_col]; start.make_start()
            end   = grid[e_row][e_col]; end.make_end()

        run = True
        world_rect = world_surf.get_rect()

        while run:
            draw_world(world_surf, grid, ROWS, COLS, cell_size)
            world_rect = render(world_surf, win, zoom_factor, cam_offset_x, cam_offset_y)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and start and end:
                        def draw_step():
                            draw_world(world_surf, grid, ROWS, COLS, cell_size)
                            render(world_surf, win, zoom_factor, cam_offset_x, cam_offset_y)

                        for row in grid:
                            for spot in row:
                                spot.update_neighbors(grid)

                        a_star_algorithm(draw_step, grid, start, end, use_zoom=True)

                    if event.key == pygame.K_s:
                        save_map_to_txt(grid, ROWS, COLS, "map.txt")
                        print("Map saved to map.txt")

                    if event.key == pygame.K_c:
                        start = None
                        end = None
                        grid = make_grid_zoom(ROWS, COLS, cell_size)

                    if event.key == pygame.K_q:
                        zoom_factor = min(MAX_ZOOM, zoom_factor + ZOOM_STEP)
                    if event.key == pygame.K_e:
                        zoom_factor = max(MIN_ZOOM, zoom_factor - ZOOM_STEP)
                    if event.key == pygame.K_LEFT:
                        cam_offset_x += CAM_SPEED
                    if event.key == pygame.K_RIGHT:
                        cam_offset_x -= CAM_SPEED
                    if event.key == pygame.K_UP:
                        cam_offset_y += CAM_SPEED
                    if event.key == pygame.K_DOWN:
                        cam_offset_y -= CAM_SPEED

                if pygame.mouse.get_pressed()[0]:
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_pos_zoom(pos, zoom_factor, world_rect, cell_size)
                    if row is None:
                        continue
                    if 0 <= row < ROWS and 0 <= col < COLS:
                        spot = grid[row][col]
                        if not start and spot != end:
                            start = spot; start.make_start()
                        elif not end and spot != start:
                            end = spot; end.make_end()
                        elif spot != end and spot != start:
                            spot.make_barrier()

                elif pygame.mouse.get_pressed()[2]:
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_pos_zoom(pos, zoom_factor, world_rect, cell_size)
                    if row is None:
                        continue
                    if 0 <= row < ROWS and 0 <= col < COLS:
                        spot = grid[row][col]
                        spot.reset()
                        if spot == start: start = None
                        elif spot == end: end = None

    pygame.quit()

main(WIN, WIDTH, HEIGHT)
