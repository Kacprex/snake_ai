import pygame, sys, time, random

# --- Config ---
TILE_SIZE = 25
GRID_WIDTH, GRID_HEIGHT = 20, 15
FPS = 10
HUD_HEIGHT = 40

# Colors
LIGHT_GREEN, DARK_GREEN = (170, 215, 81), (162, 209, 73)
BLUE, RED, WHITE, BLACK = (0, 100, 255), (220, 30, 30), (255, 255, 255), (0, 0, 0)
pygame.init()
FONT = pygame.font.SysFont("arial", 24, bold=True)

class SnakeGame:
    def __init__(self, w=GRID_WIDTH, h=GRID_HEIGHT):
        self.w, self.h = w, h
        self.screen = pygame.display.set_mode((w*TILE_SIZE, h*TILE_SIZE+HUD_HEIGHT))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = (1, 0)
        self.pending_direction = self.direction
        self.snake = [(self.w//2, self.h//2)]
        self.spawn_food()
        self.start_time = None
        self.running = False
        return self.snake

    def spawn_food(self):
        while True:
            fx, fy = random.randint(0,self.w-1), random.randint(0,self.h-1)
            if (fx,fy) not in self.snake:
                self.food = (fx,fy); break

    def change_direction(self, d):
        dx,dy=d; cur_dx,cur_dy=self.direction
        if (dx,dy)!=(-cur_dx,-cur_dy):
            self.pending_direction=d
            if not self.running: self.running, self.start_time=True,time.time()

    def update(self):
        if not self.running: return True
        self.direction=self.pending_direction
        head_x,head_y=self.snake[0]; dx,dy=self.direction
        new_head=(head_x+dx, head_y+dy)
        if (new_head[0]<0 or new_head[0]>=self.w or
            new_head[1]<0 or new_head[1]>=self.h or
            new_head in self.snake): return False
        self.snake.insert(0,new_head)
        if new_head==self.food: self.spawn_food()
        else: self.snake.pop()
        return True

    def render(self):
        # Background
        for y in range(self.h):
            for x in range(self.w):
                rect=(x*TILE_SIZE, y*TILE_SIZE+HUD_HEIGHT, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(self.screen, LIGHT_GREEN if (x+y)%2==0 else DARK_GREEN, rect)

        # HUD
        pygame.draw.rect(self.screen, BLACK, (0,0,self.w*TILE_SIZE,HUD_HEIGHT))
        elapsed=0 if self.start_time is None else int(time.time()-self.start_time)
        self.screen.blit(FONT.render(f"Time: {elapsed}",True,WHITE),(10,8))
        score=FONT.render(f"Length: {len(self.snake)}",True,WHITE)
        self.screen.blit(score,(self.w*TILE_SIZE-score.get_width()-10,8))

        # Snake
        for i,(x,y) in enumerate(self.snake):
            rect=(x*TILE_SIZE,y*TILE_SIZE+HUD_HEIGHT,TILE_SIZE,TILE_SIZE)
            pygame.draw.rect(self.screen,BLUE,rect)
            if i==0:
                eye_size=TILE_SIZE//5; off=TILE_SIZE//4
                pygame.draw.circle(self.screen,WHITE,(rect[0]+off,rect[1]+off),eye_size)
                pygame.draw.circle(self.screen,WHITE,(rect[0]+TILE_SIZE-off,rect[1]+off),eye_size)

        # Food
        fx,fy=self.food
        pygame.draw.circle(self.screen,RED,(fx*TILE_SIZE+TILE_SIZE//2,fy*TILE_SIZE+TILE_SIZE//2+HUD_HEIGHT),TILE_SIZE//2)

        pygame.display.flip(); self.clock.tick(FPS)

if __name__=="__main__":
    game=SnakeGame(); running=True
    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: pygame.quit(); sys.exit()
            elif e.type==pygame.KEYDOWN:
                if e.key in [pygame.K_LEFT,pygame.K_a]: game.change_direction((-1,0))
                if e.key in [pygame.K_RIGHT,pygame.K_d]: game.change_direction((1,0))
                if e.key in [pygame.K_UP,pygame.K_w]: game.change_direction((0,-1))
                if e.key in [pygame.K_DOWN,pygame.K_s]: game.change_direction((0,1))
        if not game.update(): print("Game Over! Len:",len(game.snake)); running=False
        else: game.render()
