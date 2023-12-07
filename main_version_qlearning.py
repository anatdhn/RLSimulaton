
import csv
import matplotlib.pyplot as plt
import numpy as np


N = 4#dim of space

# actions according to research book (0, 1, 2, 3, 4, 5)
actions = ['up', 'down', 'right', 'left', 'forward', 'backward']

# q_table = S*a == Q(s,a), S = # of cubes = 2N*2N*N,
q_table = np.zeros((N*2, N*2, N, len(actions)))#initialize q table
print(q_table)

# feedback - x,y,z,pop,m1-5
feedback = np.zeros((N*2, N*2, N, 6))#simulation table



def plotModelArg(header, table, w, h, d):#function for plotting sum of bubbles
    """Use contourf to plot cube marginals"""

    x = np.indices(table.shape)[0]
    y = np.indices(table.shape)[1]
    z = np.indices(table.shape)[2]
    col = table.flatten()
    sumb = 0
    for num in col:
        sumb += num

    print(sumb)
    norm = np.linalg.norm(col)

    fig = plt.figure()
    ax3D = fig.add_subplot(projection='3d')
    plt.xlabel("X")
    plt.ylabel("Z")

    plt.title(header)
 #   p3d = ax3D.scatter3D(x, y, z, c=col, cmap="rainbow")
    p3d = ax3D.scatter3D(x, z, y, c=col, cmap="rainbow")
    clb = plt.colorbar(p3d)
    clb.ax.set_title("Balloon shows")
    plt.show()

def plotModel():#function for plotting nax Q value for each qube
    """Use contourf to plot cube marginals"""
    data = np.zeros((N*2, N*2, N))
    for i in range(0, N*2):
        for j in range(0, N*2):
            for k in range(0, N):
                data[i, j, k] = q_table[i, j, k, np.argmax(q_table[i, j, k])]
    x = np.indices(data.shape)[0]
    y = np.indices(data.shape)[1]
    z = np.indices(data.shape)[2]
    col = data.flatten()
    norm = np.linalg.norm(col)
    fig = plt.figure()
    ax3D = fig.add_subplot(projection='3d')
    plt.xlabel("X")
    plt.ylabel("Z")
 #   plt.zlabel("Y")
  #  plt.ylabel("common Y")
    plt.title("Overall max q-value for each state (cube in space)")
 #   p3d = ax3D.scatter3D(x, y, z, c=col, cmap="rainbow")
    p3d = ax3D.scatter3D(x, z, y, c=col, cmap="rainbow")
    plt.colorbar(p3d)
    plt.show()


def episodePlod(reward_y, episode_x):
    # plotting the points
    plt.plot(episode_x, reward_y)

    # naming the x axis
    plt.xlabel('episode')
    # naming the y axis
    plt.ylabel('reward')

    # giving a title to my graph
    plt.title('reward over episodes')

    # function to show the plot
    plt.show()

# open and make new csv file to save algorithm history
def makeCsvFile():
    header = ['episode', 'batch', 'x', 'y', 'z', 'action', 'random']
    writer=None
    try:
        writer = csv.writer(open('qllog.csv','w',encoding='UTF8', newline='\n'))
    except:
        print("Error occure while trying to open/create csv file")
    writer.writerow(header)
    return writer


# get_starting_location
#   get the location where q(s,_) is the max, which will make the agent get maximum reward
def get_starting_location():
    index = np.unravel_index(np.argmax(q_table, axis=None), q_table.shape)
    return index[0], index[1], index[2]

#easy-negative, didnt pop 3, difficult positive ~8
def get_reward(row_index, column_index, depth_index):#fuction R performance
    movement = feedback[row_index, column_index, depth_index, 1]
    movement += feedback[row_index, column_index, depth_index, 2]
    movement += feedback[row_index, column_index, depth_index, 3]
    movement += feedback[row_index, column_index, depth_index, 4]
    movement /= 4
    print(movement)
    # didn't pop /
    if feedback[row_index, column_index, depth_index, 0] == 0:   
      return 0.2
    #easy
    elif movement >=0.8:
        return -0.3
    else:     # pop -not easy - by preformance
        return 1-movement*0.5  #average m around 1-easy- negative, average m around 


def get_next_action(current_row_index, current_column_index, current_depth_index):

    global epsilon
    action_probabilities = np.ones(6, dtype=float) * epsilon / 6
    best_action = np.argmax(q_table[current_row_index, current_column_index, current_depth_index])
    action_probabilities[best_action] += (1.0 - epsilon)
    print (action_probabilities)

    # ['up', 'down', 'right', 'left', 'forward', 'backward']

    flag = 1
    while flag:
        action = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)
        if action == 0 and current_column_index != (N*2)-1:
            flag = 0
        elif action == 1 and current_column_index != 0:
            flag = 0
        elif action == 2 and current_row_index != (N*2)-1:
            flag = 0
        elif action == 3 and current_row_index != 0:
            flag = 0
        elif action == 4 and current_depth_index != N-1:
            flag = 0
        elif action == 5 and current_depth_index != 0:
            flag = 0

    return action


def get_next_location(current_row_index, current_column_index, current_depth_index, action_index):#function R action
    new_row_index = current_row_index
    new_column_index = current_column_index
    new_depth_index = current_depth_index
    if actions[action_index] == 'up':
        new_column_index += 1
    if actions[action_index] == 'down':
        new_column_index -= 1
    if actions[action_index] == 'right':
        new_row_index += 1
    if actions[action_index] == 'left':
        new_row_index -= 1
    if actions[action_index] == 'forward':
        new_depth_index += 1
    if actions[action_index] == 'backward':
        new_depth_index -= 1
    return new_row_index, new_column_index, new_depth_index


# write log to csv file
def log(writer, episode, batch, row_index, column_index, depth_index, action_index, isRandom):
    str = ["" for x in range(7)]
    str[0] = episode
    str[1] = batch
    str[2] = row_index
    str[3] = column_index
    str[4] = depth_index
    str[5] = actions[action_index]
    str[6] = isRandom
    writer.writerow(str)


def get_feedback(episode):
#    if episode >= 6:
 #       episode = 4
    with open('vrsim.csv', 'r') as fp:
        reader = csv.reader(fp)
        for row in list(reader)[1:]:

            if int(row[0]) == episode%10+1:
                feedback[int(row[1]), int(row[2]), int(row[3]), 0] = int(row[4])
                feedback[int(row[1]), int(row[2]), int(row[3]), 1] = float(row[5])
                feedback[int(row[1]), int(row[2]), int(row[3]), 2] = float(row[6])
                feedback[int(row[1]), int(row[2]), int(row[3]), 3] = float(row[7])
                feedback[int(row[1]), int(row[2]), int(row[3]), 4] = float(row[8])
            #   feedback[int(row[1]), int(row[2]), int(row[3]), 5] = float(row[9])


"""#### Train the AI Agent using Q-Learning"""

# define training parameters
discount_factor = 1
learning_rate = 0.4
ExplorationRate = 10000
epsilon = 0.9
balloon_daily = np.zeros((N*2, N*2, N))
balloon_total = np.zeros((N*2, N*2, N))

def start(writer):
    global q_table, balloon_daily, epsilon
    print(q_table)
    reward_y = list(range(1, 11))
    episode_x = list(range(1, 11))
    total_reward = 0
    total_actions = 0
    last_reward = 0
    grade = 0
    zone1reward=0
    zone2reward=0
    zone3reward=0
             
    zone1ballons=0
    zone2ballons=0
    zone3ballons=0
    # run through 10 episodes
    for episode in range(10):
        # epsilon = epsilon*(1-(episode/100))
        episode_reward = 0
        get_feedback(episode)
        # get the starting location for this episode
        row_index, column_index, depth_index = get_starting_location()
        # for each episode (day), we train 1000 times:
        for batch in range(400):

            action_index = get_next_action(row_index, column_index, depth_index)
            with open("Steps.csv","a+") as steps_csv: #
                tmpWriter = csv.writer(steps_csv,delimiter=',')
                tmpWriter.writerow([episode,batch,row_index, column_index, depth_index])
                

            # perform the chosen action, and transition to the next state (i.e., move to the next location)
            old_row_index, old_column_index, old_depth_index = row_index, column_index, depth_index  # store the old row and column indexes
            row_index, column_index, depth_index = get_next_location(row_index, column_index, depth_index, action_index)
            total_actions += 1

            # receive the reward for moving to the new state, and calculate the temporal difference
            reward = get_reward(row_index, column_index, depth_index)

            old_q_value = q_table[old_row_index, old_column_index, old_depth_index, action_index]
            temporal_difference = reward + (discount_factor * np.max(q_table[row_index, column_index, depth_index])) - old_q_value

            # update the Q-value for the previous state and action pair(Rnew)
            new_q_value = (1-learning_rate)*old_q_value + (learning_rate * temporal_difference)
            q_table[old_row_index, old_column_index, old_depth_index, action_index] = new_q_value

            balloon_daily[row_index, column_index, depth_index] += 1
            balloon_total[row_index, column_index, depth_index] += 1
            episode_reward += reward
            total_reward += reward
           # epsilon = max(min(ExplorationRate/(total_actions*total_reward), 0.9), 0.8)  #anat from 0.3 to 0.8
            epsilon = max(min(ExplorationRate/(total_actions*total_reward), 0.5), 0.3)
            print('EPSILON')
            print(ExplorationRate/(total_actions*total_reward))
            print(epsilon)
            if ((row_index==6)|(row_index==7))&((column_index==6)|(column_index==7))&((depth_index==2)|(depth_index==3)):
                zone1ballons+=1
                zone1reward+=reward
            elif ((row_index==6)|(row_index==7))&((column_index==5))&((depth_index==2)|(depth_index==3)):
                zone2ballons+=1
                zone2reward+=reward
            elif ((row_index==6)|(row_index==7))&((column_index==6)|(column_index==7))&((depth_index==1)):
                zone2ballons+=1
                zone2reward+=reward
            elif ((row_index==5))&((column_index==6)|(column_index==7))&((depth_index==2)|(depth_index==3)):
                zone2ballons+=1
                zone2reward+=reward
            else:
                zone3ballons+=1
                zone3reward+=reward
                
          

        reward_y[episode] = episode_reward
        plotModelArg("Day " + str(episode+1), balloon_daily, N * 2, N * 2, N)
        #write balloon_daily
        print(balloon_daily)
        print("Day " + str(episode+1))
        print(zone1ballons)
        print(zone2ballons)
        print(zone3ballons)
        
        with open("Day " + str(episode+1)+".csv","w+") as day_csv:
            csvWriter = csv.writer(day_csv,delimiter=',')
            for i in range(0,8):
                for j in range(0,8):
                    for k in range(0,4):
                      #  print([i,j,k,balloon_daily[i, j, k]])
                        csvWriter.writerow([i,j,k,balloon_daily[i, j, k]])
   #         csvWriter.writerows(balloon_daily)
        plotModel()
        with open("Day_RES " + str(episode+1)+".csv","w+") as dayres_csv:
            csvresWriter = csv.writer(dayres_csv,delimiter=',')
            csvresWriter.writerow([zone1ballons,zone1reward])
            csvresWriter.writerow([zone2ballons,zone2reward])
            csvresWriter.writerow([zone3ballons,zone3reward])
        
        zone1reward=0
        zone2reward=0
        zone3reward=0
             
        zone1ballons=0
        zone2ballons=0
        zone3ballons=0    
            
        balloon_daily = np.zeros((N * 2, N * 2, N))

    episodePlod(reward_y, episode_x)
    plotModelArg("Total balloons show", balloon_total, N * 2, N * 2, N)



def main():
    writer = makeCsvFile()
#define -100 value for ilegitimite actions
    for i in range(0, N*2):
        for j in range(0, N):
            q_table[0, i, j, 3] = -100
            q_table[i, 0, j, 1] = -100
            q_table[N*2-1, i, j, 2] = -100
            q_table[i, N*2-1, j, 0] = -100

    for i in range(0, N*2):
        for j in range(0, N*2):
            q_table[i, j, 0, 5] = -100
            q_table[i, j, N-1, 4] = -100

    global discount_factor, learning_rate

    start(writer)
    plotModel()


if __name__ == "__main__":
    main()
