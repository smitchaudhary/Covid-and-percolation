# Fixed hyperparameters
L = 300 # Delft population: 90000
mobility_factor = 0.001  # Probability of a person of having a long range interaction in a day
recovery_period = 14    # Days it takes to a person to heal from the disease on average
init_infected_ratio = 2*10**(-4)   # Ratio of initially infected probability (around 20 people for 90000 people)
total_death_prob = 0.005 # Probability of a person of dying because of the disease WITH ACCESS TO HEALTHCARE
max_death_prob   = 5*total_death_prob

# Disease hyperparameters
inf_prob = 0.055 # Infection probability with no lockdown
visual_bool = False

# Health care system parameters
icu_beds_percent = 6e-5 # Number of ICU beds per 100 inhabitants
hospilatization_risk = 0.005 # Probability of a person of being hospilased from the disease

#Lockdown parameters
lockdown_threshold = 0.0018 # Ratio of infected people needed to impose lockdown
lifting_lockdown_threshold = lockdown_threshold * 0.1
lockdown_mobility_factor = mobility_factor*0 # Mobility factor

#inferred hyperparameters
daily_death_prob = 1 - (1 - total_death_prob)**(1/recovery_period)
daily_max_death_prob = 1 - (1 - max_death_prob)**(1/recovery_period)

#Vaccination
vaccination_rate = 1/450
vaccine_efficiency = 0.9
vaccine_dev_time = 255
