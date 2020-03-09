import numpy as np
import matplotlib.pyplot as plt
import random as rand
import time

# Here we will be defining some constants:
radii = [0.5, 1.0]    # 0: regular nut, 1: Brazilian nut
h_allocation = 1.0    # Distance between each nut in the set-up
box_w = 5.0          # Width of the box

def display_box(nuts, positions, show_axis):
    """
    Displays the box of nuts.
    
    Args:
        nuts: array describing the type of every nut. 0: regular 1: brazilian.
        positions: 2D array describing the x and y coordinate of every nut.
        show_axis: bool on whether or not to display labels and values in the axes.
        
    Returns:
        The figure
    """
    circle_list = []
    for i in range(len(nuts)):
        if nuts[i] == 1:
            new_circle = plt.Circle(positions[i], radii[1], color='r')
            circle_list.append(new_circle)
        elif nuts[i] == 0:
            new_circle = plt.Circle(positions[i], radii[0], color='r')
            circle_list.append(new_circle)

    fig, ax = plt.subplots(figsize=(5, 10))
    
    ax.set_xlim([0, box_w])
    ax.set_ylim([0, 10])
    
    
    for drawn_circle in circle_list:
        ax.add_artist(drawn_circle)
    return fig
    
def initial_preparation(max_n, b_nut_chance):
    """
    Prepares the initial set of nuts according to the document.
    Always starts with a brazilian nut at the bottom.
    
    Args:
        max_n: maximum number of nuts
        b_nut_chance: chance for a nut to be brazilian
    
    Returns: 
        nuts: array of the types of nuts generated. 1 is brazilian, 0 is regular
        positions: 2D array of the locations of the nuts
    """
    nuts = np.zeros(max_n)
    positions = np.zeros((max_n,2))
    
    current_h = radii[1]
    nuts[0] = 1
    positions[0] = np.array([rand.randrange(radii[1], box_w-radii[1]), current_h])
    n = 1
    current_h = radii[0] + radii[1]

    
    while n < max_n:
        current_h += 2*radii[0]
        nuts[n] = 0
        positions[n] = np.array([rand.uniform(radii[0], box_w-radii[0]), current_h])
        n += 1
    return nuts, positions

def check_overlap_walls(nut, position):
    """
    Checks if a nut in a given position overlaps with the bottom or the sides of the box
    
    Args: 
        nut: type of the nut
        position: array of the x and y coordinates of the nut
    Returns:
        bool True if the nut overlaps with the walls of the box
    """
    if position[0] >= radii[int(nut)] and position[0] <= box_w-radii[int(nut)] and position[1] > radii[int(nut)]:
        return False
    else: 
        return True

def check_overlap_nuts(nut1, position1, nut2, position2):
    """
    Checks if two nuts in the given positions overlap with each other 
    
    Args: 
        nut1: type of the first nut
        position1: array of the x and y coordinates of the first nut 
        nut2: type of the second nut
        position2: array of the x and y coordinates of the second nut 
    """
    if np.sqrt((position1[0]-position2[0])**2 + (position1[1]-position2[1])**2) > (radii[int(nut1)]+radii[int(nut2)]):
        return False
    else: 
        return True

def nearest_neighbor(c_n, nuts, positions):
    """ 
    Returns the nearest neighbor between the current nut and the set of nuts. 

    Args:
        c_n: index of the nut to be evaluated
        nuts: array containing all the nut types
        positions: array containing the positions of all the nuts
    """

    nearest_neighbor = None
    nearest_neighbor_nut = None
    assert len(nuts) == len(positions), "Nut types and positions arrays should be the same length"
    for n in range(len(nuts)):
        d = np.sqrt((positions[n][0]-positions[c_n][0]-radii[int(nuts[n])]-radii[int(nuts[c_n])])**2 \
                  + (positions[n][1]-positions[c_n][1]-radii[int(nuts[n])]-radii[int(nuts[c_n])])**2)
        if (nearest_neighbor == None) or (d > 0.0 and d < nearest_neighbor) :
            nearest_neighbor = d
            nearest_neighbor_nut = n
    return nearest_neighbor, c_n, nearest_neighbor_nut
     
def largest_nearest_neighbor(nuts, positions):
    """
    Returns the largest possible shortest distance between the given array of particles.
    
    Args:
        nuts: array containing the types of nuts
        positions: array containing the x and y corrdinates of the positions of the nuts
    Returns:
        largest_distance: largest distance found
        a: index of the first nut
        b: index of the second nut
    """
    largest_distance = 0
    a = 0
    b = 0
    for i in range(len(nuts)):
        value, n1, n2 = nearest_neighbor(i, nuts, positions)
        if value > largest_distance:
            a = n1
            b = n2
            largest_distance = value

    return largest_distance, a, b

def energy(positions):
    """
    Calculates E/mg of the system given the positions of the nuts
    
    Args: 
        positions: x and y coordinates of the nuts
    Returns:
        Sum of E/mg for all the nuts in the system 
    """
    energy = 0
    for position in positions:
        energy += position[1]
    return energy

def shake_box(positions):
    """
    Shakes the box, multiplying all the vertical coordinates by 2

    Args: 
        positions: positions to be shook
    Returns:
        _Shaken_ positions
    """
    new_positions = np.zeros(positions.shape)
    for i in range(len(new_positions)):
        new_positions[i] = np.array([positions[i][0], 2*positions[i][1]])
    return new_positions

def mc_method(nuts, positions, beta_value, d_e_minimum):
    """
    Metropoli Monte Carlo method to _settle_ a set of nuts.
    
    Runs until after the difference in energy after a 1000 attempted moves is lower than the threshold. 
    **Note that we assume that all the nuts are the same weight.**
    
    Input: 
    
    Returns:
    
    """
    current_positions = np.copy(positions)
    
    # We will exit this loop once d_e > d_e_minimum
    while True:
        attempts = 0 
        d_e = 0
        max_step_length, nut1, nut2 = largest_nearest_neighbor(nuts,current_positions)

        while attempts < 1000:
            current = rand.randrange(max_n)

            attempts += 1
            step = rand.uniform(0, max_step_length)
            angle = rand.uniform(0, 2*np.pi)
            x_step = step*np.cos(angle)
            y_step = step*np.sin(angle)

            
            # Check if the change in energy is positive (i.e. positive y_step)
            if y_step > 0:
                prob = np.exp(- beta_value*y_step*9.81)
                if rand.uniform(0,1) > prob: # Now we will make the attempt pass the probability test
                    continue
            
            new_positions = np.copy(current_positions)
            new_nuts = np.copy(nuts)
            
            new_positions[current] = [new_positions[current][0]+x_step, new_positions[current][1]+y_step]
            nuts_to_check = np.delete(new_nuts, current)
            positions_to_check = np.delete(new_positions,current, axis=0)

            # Testing for overlap with walls
            if check_overlap_walls(nuts[current], new_positions[current]):
                continue

            # Testing for collisions
            passed_collisions = True
            for i in range(len(nuts_to_check)):
                if check_overlap_nuts(nuts_to_check[i], positions_to_check[i], nuts[current], new_positions[current]):
                    passed_collisions = False
                    break
            if not passed_collisions:
                continue

            d_e += y_step*9.81
            
            current_positions = new_positions
            #fig1 = display_box(nuts, current_positions,0)
            #plt.savefig("final_config.png")
            #plt.close(fig1)
            
            
            # If we moved either of the nuts involved in the largest nearest neighbor value, recompute it
            if current == nut1 or current == nut2:
                max_step_length, nut1, nut2 = largest_nearest_neighbor(nuts, current_positions)
    
        if d_e > d_e_minimum:
            break
                
    return nuts, current_positions

# Actual program
start = time.time()

max_n = 50
nuts, positions = initial_preparation(max_n, 0)
fig1 = display_box(nuts, positions,0)
plt.savefig("initial_config.png")
positions_after_shake = np.copy(positions)

shake_iters = 10
for i in range(shake_iters):
    next_nuts, positions_after_shake = mc_method(nuts, positions_after_shake, 10**16, -1)
    if i is not shake_iters-1:
        positions_after_shake = shake_box(positions_after_shake)

    
fig2 = display_box(nuts, positions_after_shake,0)
plt.savefig("final_config.png")

end = time.time()
print(end - start, "s")
