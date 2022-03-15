import numpy as np
import random
from config import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
from random import choices

#np.random.seed(0)
#random.seed(0)

class Simulation():

    def __init__(self, L , inf_prob = inf_prob, daily_death_prob = daily_death_prob, daily_max_death_prob = daily_max_death_prob , mobility_factor = mobility_factor, vac_rate = vaccination_rate):
        '''
        Initialize the simulation.
        Parameters:
            L : int
                Size of the lattice
            inf_prob : float
                Infection probability. Models how easily the virus is spread.
            daily_death_prob : float
                Probability of a person who is suffering from the virus dieing in the net day.
            daily_max_death_prob : float
                Maximum allowed death probability even when healthcare is overburdened.
            mobility_factor : float
                Models how far the virus is spread beyond the nearest neighbours.
        '''
        self.L = L
        self.population = L**2
        self.icu_beds = icu_beds_percent*self.population
        self.healthcare_threshold = self.icu_beds/ hospilatization_risk
        self.inf_prob = inf_prob
        self.lattice = [[None]*L for i in range(L)]
        self.infected_people = []
        self.daily_death_prob = daily_death_prob
        self.mobility_factor = mobility_factor
        self.daily_max_death_prob = daily_max_death_prob
        self.todays_death_prob = daily_death_prob
        self.initialize_lattice()
        self.infect_rand()

        self.n_infected_people_t  = [len(self.infected_people)]
        self.n_recovered_people_t = [0]
        self.n_deceased_people_t = [0]

        self.lockdown_status = False
        self.todays_mobility_factor = mobility_factor
        self.todays_inf_prob = inf_prob
        self.day = 0
        self.vaccination_rate = vac_rate

    def run_simulation(self, visualization = visual_bool):
        '''
        Runs the simulation until no person is an active case of infection.

        Parameters:
            visualization_bool: bool
                Allows for a visual simulation. Visualization shows infected, recovered, deceased and susceptible status of each person.
        '''
        if visualization:
            #colormap from https://stackoverflow.com/a/60870122
            col_dict={-2:"black",
                    -1:"red",
                    0:"white",
                    1:"blue"}

            # We create a colormar from our list of colors
            cm = ListedColormap([col_dict[x] for x in col_dict.keys()])

            labels = np.array(["Deceased","Infected","Susceptible","Recovered \n or Vaccinated"])
            len_lab = len(labels)

            # prepare normalizer
            ## Prepare bins for the normalizer
            norm_bins = np.sort([*col_dict.keys()]) + 0.5
            norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
            ## Make normalizer and formatter
            norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
            fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

            # Plot our figure
            fig,ax = plt.subplots()
            im = ax.imshow(self.get_lattice_status(), cmap=cm, norm=norm)

            diff = norm_bins[1:] - norm_bins[:-1]
            tickz = norm_bins[:-1] + diff / 2
            cb = fig.colorbar(im, format=fmt, ticks=tickz)

            while len(self.infected_people) != 0 and len(self.infected_people) != self.population:
                self.update()
                im.set_data(self.get_lattice_status())
                if self.day < vaccine_dev_time:
                    plt.title(f'Day {self.day} \n Percentage of infected people is {np.round(100*len(self.infected_people)/self.population, 1)} % \n Vaccination not started yet')
                else:
                    plt.title(f'Day {self.day} \n Percentage of infected people is {np.round(100*len(self.infected_people)/self.population, 1)} % \n Vaccination started')
                plt.draw()
                plt.pause(1e-6)
                #Uncomment to create gif file afterwards
                #plt.savefig(f'Figures/{self.day}.png')
            plt.show()
        else:
            while len(self.infected_people) != 0 and len(self.infected_people) != self.population:
                self.update()

    def initialize_lattice(self):
        '''
        Initialize the lattice list of lists with its elements as Person objects.
        '''
        for i in range(self.L):
            for j in range(self.L):
                self.lattice[i][j] = Person( (i,j) )

    def infect_rand(self, init_infected_ratio = init_infected_ratio):
        '''
        Randomly infect some people to start the simulation.
        Parameters:
            init_infected_ratio = float
                Ratio of the total population initially infected.
                If no init_infected_ratio is given or given as 0, then only one person will be infected.
        '''
        #random.seed(0)
        num_infections = max(1, int((self.population) * init_infected_ratio))

        flatten = lambda t: [item for sublist in t for item in sublist] #expression that flattens list
        flattened_lattice = flatten(self.lattice)
        init_inf = random.sample(  flattened_lattice, num_infections)
        for p in init_inf:
            p.change_inf_status(-1)
            self.infected_people.append(p)

    def get_lattice_status(self):
        '''
        Returns a numpy array for the lattice with:
         -2 for deceased people
         -1 for infected people
         0 for uninfected people
         1 for recovered people
        '''
        status = np.zeros((self.L,self.L))
        for i in range(self.L):
            for j in range(self.L):
                status[i,j] = self.lattice[i][j].inf_status
        return status

    def update(self):
        '''
        Updates the lattice for a single time step.
        '''
        recov_people = 0
        deceased_people = 0

        # Lockdown
        self.lockdown()

        # Vaccinations
        self.vaccinate()

        # If there are more hospitalized people than ICU beds

        self.todays_death_prob = self.daily_death_prob

        x = max(0,  (self.n_infected_people_t[-1] - self.healthcare_threshold)/self.healthcare_threshold )
        self.todays_death_prob += self.todays_death_prob*x
        self.todays_death_prob = min(self.todays_death_prob , self.daily_max_death_prob)

        for inf_per in self.infected_people:

            nn = self.near_neigh(inf_per)

            # Long range interactions
            while True:
                if np.random.random() < self.todays_mobility_factor:
                    nn.append(self.lattice[np.random.randint(0,self.L)][np.random.randint(0,self.L)])
                else:
                    break


            # Infection of contacts
            for p in nn:
                if p.inf_status == 0 and np.random.rand() < self.todays_inf_prob:
                    p.change_inf_status(-1)
                    self.infected_people.append(p)

            inf_per.days_to_heal -= 1

            # Deceasing
            if np.random.random() < self.todays_death_prob:
                inf_per.change_inf_status(-2)
                self.infected_people.remove(inf_per)
                deceased_people += 1

            # Healing
            elif inf_per.days_to_heal == 0 :
                inf_per.change_inf_status(1)
                self.infected_people.remove(inf_per)
                recov_people += 1

        self.n_infected_people_t.append( len(self.infected_people) )
        self.n_recovered_people_t.append(  self.n_recovered_people_t[-1] + recov_people  )
        self.n_deceased_people_t.append(  self.n_deceased_people_t[-1] + deceased_people  )
        self.day += 1

    def lockdown(self):
        '''
        Impose lockdown if infection rise above a certian threshold and lift it if they fall below another threshold.
        Imposing lockdown reduces mobility of the people.
        '''

        if self.lockdown_status == True:
            if self.n_infected_people_t[-1]/self.population < lifting_lockdown_threshold:
                self.todays_mobility_factor = self.mobility_factor
                self.lockdown_status = False
                print(f'Lifting lockdown')
        else:
            if self.n_infected_people_t[-1]/self.population > lockdown_threshold:
                self.todays_mobility_factor = lockdown_mobility_factor
                self.lockdown_status = True
                print(f'Imposing lockdown')

    def vaccinate(self):
        '''
        Vaccinate a percentage of population once the vaccine has been developed.
        '''
        if self.day > vaccine_dev_time:
            flatten = lambda t: [item for sublist in t for item in sublist] #expression that flattens list
            flattened_lattice = flatten(self.lattice)
            to_vaccinate = choices(flattened_lattice, k = int(self.population*self.vaccination_rate))
            for per in to_vaccinate:
                if per.inf_status == 0 and np.random.random() < vaccine_efficiency:
                    per.inf_status = 1

    def near_neigh(self, p):
        '''
        Returns a list of Person objects that are the nearest neighbours of another Person.
        Parameters:
            p: Person object
                Person of whom the nearest neighbours will be returned.
        '''

        pos_i, pos_j = p.position[0], p.position[1]
        near_neigh = [ self.lattice[ pos_i ][ (pos_j + 1)%self.L ],\
        self.lattice[(pos_i-1)%self.L][(pos_j )%self.L],\
        self.lattice[(pos_i+1) %self.L][ pos_j],\
        self.lattice[pos_i][(pos_j - 1)%self.L] ]

        if self.lockdown_status == True:
            if np.random.random() < 0.45:
                near_neigh = choices( near_neigh, k = 3 )
            else:
                near_neigh = choices( near_neigh, k = 4 )
        return near_neigh

    def epidemic_curves(self, plot_bool = False, name = 'name', rate = 0):
        '''
        Plot various relevant quantites : Infected population etc.

        Parameters:
            plot_bool : bool
                Plot the curves only if plot_bool is True. False by default.
        '''
        n_recovered = np.array(self.n_recovered_people_t)
        n_infected = np.array(self.n_infected_people_t)
        n_deceased = np.array(self.n_deceased_people_t)

        # 7-days moving average
        n_infected_avr = np.convolve(n_infected, np.ones(7), 'full') / 7
        n_deceased = np.convolve(n_deceased, np.ones(7), 'full') / 7

        plt.plot(100*n_infected_avr/self.population, label = f'{np.round(100*rate, 2)} % per day')


        if plot_bool:
            #Tweak to showcase what is needed
            plt.plot( 100*n_infected_avr/self.population, label = 'Currently Infected People')
            #plt.plot(n_recovered, label = 'Total Recovered People')
            #plt.plot(100*n_deceased/self.population, label = f'Total deceased People {np.round(100*n_deceased[-1]/self.population,2)}%')
            plt.axhline(y=100*self.healthcare_threshold/self.population, color='r', linestyle='-', label = 'Healthcare threshold')
            plt.axhline(y=100*lockdown_threshold, color='g', linestyle='-', label = 'Lockdown imposing threshold')
            plt.axhline(y=100*lifting_lockdown_threshold, color='b', linestyle='-', label = 'Lockdown lifting threshold')
            plt.axvline(x=vaccine_dev_time, color = 'y', linestyle = '-', label = 'Vaccination starts')
            plt.title(f"Epidemic curves with lockdown and vaccination. {np.round( 100*(self.n_deceased_people_t[-1] + self.n_recovered_people_t[-1])/self.population, 1)}% of the population were infected")
            plt.xlabel('Day')
            plt.ylabel('Population percentage')
            plt.legend()
            plt.show()



class Person():
    '''
    Class describing a person and their infection status.
    '''
    def __init__(self, position):
        '''
        Initialize the Person instance with no infection at given position.

        Parameters:
            position : tuple (int, int)
                Position of the person in the lattice.
        '''
        self.position = position
        self.inf_status = 0
        self.days_to_heal = 0

    def change_inf_status(self, new_status):
        '''
        Changes the infection status of the person.

        Parameters:
            new_status: int
                The new infection status of the person. SIRD model
                -2 : deceased
                -1 : infected
                 0 : susceptible
                +1 : recovered
        '''

        self.inf_status = new_status
        if new_status == -1:
            self.days_to_heal = recovery_period
