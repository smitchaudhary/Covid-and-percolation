 # Project 3: Covid Simulation and percolation theory

This project aims to simulate the spread of an infectious disease such as Covid-19, caused by the SARS-CoV2 virus. This is done by means of the SIRD (Susceptible Infected Recovered Deceased) epidemiological model. People are represented by sites in a lattice, and the virus is allowed to transmit from infected to susceptible individuals. See the presentation slides in Presentation.pdf for more information.

### Current features:

1. Nearest neighbours get infected with a certain probability.
2. Infected people get healed after a recovery period.
3. Mobility models people moving around and having more conmtacts and spreading of the virus beyond the nearest neighbours.
4. The healthcare capacity is limited and when the current active cases are beyond the healthcare capacity, the risk of passing away increases without proper care.
5. Imposing and lifting lockdowns based on respective thresholds for currrent active cases. Premature lifting of thresholds causes subsequent secondary waves of infections.
6. Vaccination of the population starts after certain number of days that it takes to develop a vaccine.

## File description
- `library.py`: Code that simulates SIRD model and its extensions on a 2D lattice.
- `config.py`: Contains global default hyperparameters that can be overwritten if needed. 
- `main.py`: Serves as a template to obtain data and plots for various parameter regimes.
- `plots.py`: Allows the creation of GIF files, if one first saves the necessary plots.
- `journal.md`: Contains our progress throughout the duration of the project.

    
## Requirements
- Python 3
- numpy
- scipy
- matplotlib
- random


## Authors
- Smit Chaudhary
- Ignacio Fernández Graña
- Georgios Sotiropoulos
