import random, numpy as np

class SnakeGameAI:
    def __init__(self,w=10,h=10):
        self.w,self.h=w,h; self.reset()

    def reset(self):
        self.direction=(1,0); self.pending=self.direction
        self.snake=[(self.w//2,self.h//2)]
        self.spawn_food(); return self.get_state()

    def spawn_food(self):
        while True:
            fx,fy=random.randint(0,self.w-1),random.randint(0,self.h-1)
            if (fx,fy) not in self.snake: self.food=(fx,fy); break

    def step(self,action):
        dx,dy=self.direction
        if action==0 and dx==0: self.pending=(-1,0)
        if action==1 and dx==0: self.pending=(1,0)
        if action==2 and dy==0: self.pending=(0,-1)
        if action==3 and dy==0: self.pending=(0,1)
        self.direction=self.pending
        head_x,head_y=self.snake[0]; dx,dy=self.direction
        new_head=(head_x+dx,head_y+dy)

        reward=0.0; done=False
        if (new_head[0]<0 or new_head[0]>=self.w or
            new_head[1]<0 or new_head[1]>=self.h or
            new_head in self.snake): return self.get_state(),-10,True,len(self.snake)

        self.snake.insert(0,new_head)

        if new_head==self.food:
            reward=10; self.spawn_food()
        else:
            self.snake.pop()
            # distance shaping
            old_dist=abs(head_x-self.food[0])+abs(head_y-self.food[1])
            new_dist=abs(new_head[0]-self.food[0])+abs(new_head[1]-self.food[1])
            if new_dist<old_dist: reward+=1
            else: reward-=0.5

        return self.get_state(),reward,done,len(self.snake)

    def get_state(self):
        state=np.zeros((self.h,self.w),dtype=int)
        for x,y in self.snake: state[y,x]=1
        fx,fy=self.food; state[fy,fx]=2
        return state.flatten()
