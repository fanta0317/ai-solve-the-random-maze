import numpy as np
import tkinter as tk
import random
import time
import random as rd

def ald(grid:np.ndarray,size:int) -> np.ndarray:
    output_grid = np.empty([size*3, size*3],dtype=int)
    output_grid[:] = 1
    c = size*size # number of cells to be visited
    i = rd.randrange(size)
    j = rd.randrange(size)
    while np.count_nonzero(grid) < c:
  
        # visit this cell
        grid[i,j] = 1

        w = i*3 + 1
        k = j*3 + 1
        output_grid[w,k] = 0

        can_go = [1,1,1,1]

        if i == 0:
            can_go[0] = 0
        if i == size-1:
            can_go[2] = 0
        if j == 0:
            can_go[3] = 0
        if j == size-1:
            can_go[1] = 0
        
        # it makes sense to choose neighbour among available directions
        neighbour_idx = np.random.choice(np.nonzero(can_go)[0]) # n,e,s,w

        if neighbour_idx == 0:
            # has been visited?
            if grid[i-1,j] == 0:
                # goto n
                output_grid[w-1,k] = 0
                output_grid[w-2,k] = 0
            i -= 1
                    
        
        if neighbour_idx == 1:
            if grid[i,j+1] == 0:
                # goto e
                output_grid[w,k+1] = 0
                output_grid[w,k+2] = 0
            j += 1
          
        if neighbour_idx == 2:
            if grid[i+1,j] == 0:
                # goto s
                output_grid[w+1,k] = 0
                output_grid[w+2,k] = 0 
            i += 1
        

        if neighbour_idx == 3:
            # goto w
            if grid[i,j-1] == 0:
                output_grid[w,k-1] = 0
                output_grid[w,k-2] = 0
            j -= 1
    output_grid[1,1] = 3
    output_grid[10,10] = 2       
    return output_grid

'''def main():
    size=5

    #np.random.seed(42)
    grid = np.zeros(shape=(size,size))

    console_grid = np.array(ald(grid,size))
    print(console_grid.dtype)

    for elm in console_grid:
        maze1 =(" ".join(elm))'''









class MazeWindow:
    def __init__(self, maze):
        self.root = tk.Tk()
        #self.root.geometry('5000x5000')
        self.root.title('Maze')
        self.maze = maze
        self.labels = np.zeros(self.maze.shape).tolist()
        self.plotBackground()
    def plotBackground(self):
        for i, row in enumerate(self.maze.tolist()):
            for j, element in enumerate(row):
                bg = 'black' if element == 1 else 'red' if element == 2 else 'green' if element == 3 else 'white'
                self.labels[i][j] = tk.Label(self.root, foreground='blue', background=bg, width=2, height=1, relief='ridge', font='? 40 bold')
                self.labels[i][j].grid(row=i, column=j)
    def mainloop(self, func):
        self.root.after(1000, func)
        self.root.mainloop()
    def target(self, indexes):
        for label in [item for row in self.labels for item in row]:
            label.config(text='')
        self.labels[indexes[0]][indexes[1]].config(text = 'o')
        self.root.update()

class Agent:
    def __init__(self, maze, initState):
        self.state = initState
        self.maze = maze
        self.initQTable()
        self.actionList = ['up', 'down', 'left', 'right']
        self.actionDict = {element : index for index, element in enumerate(self.actionList)}
    def initQTable(self):
        Q = np.zeros(self.maze.shape).tolist()
        for i, row in enumerate(Q):
            for j, _ in enumerate(row):
                Q[i][j] = [0, 0, 0, 0] # up, down, left, right
        self.QTable = np.array(Q, dtype='f')
    def showQTable(self):
        for i, row in enumerate(self.QTable):
            for j, element in enumerate(row):
                print(f'({i}, {j}){element}')
    def showBestAction(self):
        for i, row in enumerate(self.QTable):
            for j, element in enumerate(row):
                Qa = element.tolist()
                action = self.actionList[Qa.index(max(Qa))] if max(Qa) != 0 else '??'
                print(f'({i}, {j}){action}', end=" ")
            print()
    def getAction(self, eGreddy=0.8):
        if random.random() > eGreddy:
            return random.choice(self.actionList)
        else:
            Qsa = self.QTable[self.state].tolist()
            return self.actionList[Qsa.index(max(Qsa))]
    def getNextMaxQ(self, state):
        return max(np.array(self.QTable[state]))
    def updateQTable(self, action, nextState, reward, lr=0.7, gamma=0.9):
        Qs = self.QTable[self.state]
        Qsa = Qs[self.actionDict[action]]
        Qs[self.actionDict[action]] = (1 - lr) * Qsa + lr * (reward + gamma *(self.getNextMaxQ(nextState)))

class Environment:
    def __init__(self):
        pass
    # Determine the result of an action in this state.
    def getNextState(self, state, action):
        row = state[0]
        column = state[1]
        if action == 'up':
            row -= 1
        elif action == 'down':
            row += 1
        elif action == 'left':
            column -= 1
        elif action == 'right':
            column += 1
        nextState = (row, column)
        try:
            # Beyond the boundary or hit the wall.
            if row < 0 or column < 0 or maze[row, column] == 1:
                return [state, False]
            # Goal
            elif maze[row, column] == 2:
                return [nextState, True]
            # Forward
            else:
                return [nextState, False]
        except IndexError as e:
            # Beyond the boundary.
            return [state, False]
    # Execute action.
    def doAction(self, state, action):
        nextState, result = self.getNextState(state, action)
        # No move
        if nextState == state:
            reward = -10
        # Goal
        elif result:
            reward = 100
        # Forward
        else:
            reward = -5
        return [reward, nextState, result]
    
def main():    
    initState = (np.where(maze==3)[0][0], np.where(maze==3)[1][0])
    # Create an Agent
    agent = Agent(maze, initState)
    # Create a game Environment
    environment = Environment()
    for j in range(0, 50):
        agent.state = initState
        m.target(agent.state)
        time.sleep(0.1)
        i = 0
        while True:
            i += 1
            # Get the next step from the Agent
            action = agent.getAction(0.9)
            # Give the action to the Environment to execute
            reward, nextState, result = environment.doAction(agent.state, action)
            # Update Q Table based on Environmnet's response
            agent.updateQTable(action, nextState, reward)
            # Agent's state changes
            agent.state = nextState
            m.target(agent.state)
            if result:
                print(f' {j+1:2d} : {i} steps to the goal.')
                break
    agent.showQTable()
    agent.showBestAction()

if __name__ == '__main__':
    #main()

    size=4

    #np.random.seed(42)
    grid = np.zeros(shape=(size,size))

    maze11111 = np.array(np.array(ald(grid,size)).tolist())

    
    lista=[]
    listc=[]
    listd=[]
    liste=[]
    listf=[]
    listg=[]
    listh=[]
    listi=[]
    listj=[]
    listk=[]
    listl=[]
    listm=[]

    listb=np.array_split(maze11111[0],12)
    for i in range(0,12):
        a=listb[i]
        b=int(a[0])
        lista.append(b)

    listb=np.array_split(maze11111[1],12)
    for i in range(0,12):
        a=listb[i]
        b=int(a[0])
        listc.append(b)

    listb=np.array_split(maze11111[2],12)
    for i in range(0,12):
        a=listb[i]
        b=int(a[0])
        listd.append(b)

    listb=np.array_split(maze11111[3],12)
    for i in range(0,12):
        a=listb[i]
        b=int(a[0])
        liste.append(b)

    listb=np.array_split(maze11111[4],12)
    for i in range(0,12):
        a=listb[i]
        b=int(a[0])
        listf.append(b)

    listb=np.array_split(maze11111[5],12)
    for i in range(0,12):
        a=listb[i]
        b=int(a[0])
        listg.append(b)

    listb=np.array_split(maze11111[6],12)
    for i in range(0,12):
        a=listb[i]
        b=int(a[0])
        listh.append(b)

    listb=np.array_split(maze11111[7],12)
    for i in range(0,12):
        a=listb[i]
        b=int(a[0])
        listi.append(b)

    listb=np.array_split(maze11111[8],12)
    for i in range(0,12):
        a=listb[i]
        b=int(a[0])
        listj.append(b)

    listb=np.array_split(maze11111[9],12)
    for i in range(0,12):
        a=listb[i]
        b=int(a[0])
        listk.append(b)

    listb=np.array_split(maze11111[10],12)
    for i in range(0,12):
        a=listb[i]
        b=int(a[0])
        listl.append(b)

    listb=np.array_split(maze11111[11],12)
    for i in range(0,12):
        a=listb[i]
        b=int(a[0])
        listm.append(b)

    maze = np.array([lista,
                    listc,
                    listd,
                    liste,
                    listf,
                    listg,
                    listh,
                    listi,
                    listj,
                    listk,
                    listl,
                    listm,
                    ])        

   
# -1 is origin, 0 is road, 1 is wall, 2 is goal 
    '''maze = np.array([
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 1, 1 ,1],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [-1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 1 ,2]
    ])'''


m = MazeWindow(maze)
m.mainloop(main)
