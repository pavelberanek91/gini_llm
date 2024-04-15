# Large language models in multi agents simulations

## Data description

* RunID = increases with steps and iterations, unique id for step in iteration
* iteration = same model runs multiple times (defined by iterations), this is id of such run
* Step = in each iteration a specified number of simulation steps is made (defined by max_steps), this is id of such steps
* AgentID = id of agent in step

Parameters that are changeable in each iterations by range (same in steps, can be different in iterations).

* n_agents = number of agents in iteration that are interacting
* space_width = width of space where agents meet, number of cells in x-axis
* space_height = height of space where agents meet, number of cells in y-axis

Calculated informations in steps.

* Gini = gini coefficient of inequality (0 = wealth is equally distributed, 1 = all the wealth holds one agent)
* Wealth = how much wealth (money, education, information) agent have at concrete step
* Steps_not_given = how many times from last transaction havent agent transfered its wealth
