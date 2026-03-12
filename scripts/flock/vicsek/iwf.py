from soc_emp.envs.flock.vicsek import Vicsek

if __name__ == '__main__':

    num_agents = 10
    grid_size = 5.0
    neighbor_radius = 0.5
    speed = 1.0
    J = 0.1
    D = 0.0
    dt = 0.05

    flock = Vicsek(num_agents, grid_size, neighbor_radius, speed, J, D, dt)
    print(flock)