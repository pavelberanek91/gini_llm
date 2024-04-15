import mesa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def agent_portrayal(agent):
    portrayal = {
        'Shape': 'circle',
        'Filled': 'true',
        'Layer': 0,
        'Color': 'red',
        'r': 0.5,
    }
    return portrayal


class MoneyAgent(mesa.Agent):
    '''An agent with fixed initial wealth.'''
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 1
        self.steps_not_given = 0

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        cellmates.pop(cellmates.index(self))
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1
            self.steps_not_given = 0
        else:
            self.steps_not_given += 1

    def step(self):
        self.move()
        if self.wealth > 0:
            self.give_money()
        else:
            self.steps_not_given += 1


class BoltzmannWealthModel(mesa.Model):
    def __init__(self, n_agents, space_width, space_height):
        '''A model with specified number of agents'''
        super().__init__()
        self.n_agents = n_agents
        self.grid = mesa.space.MultiGrid(space_width, space_height, torus=True)
        self.schedule = mesa.time.RandomActivation(self)

        #create agents in model
        for iagent in range(self.n_agents):
            agent = MoneyAgent(unique_id=iagent, model=self)
            self.schedule.add(agent)
            rand_x = self.random.randrange(self.grid.width)
            rand_y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (rand_x, rand_y))

        self.datacollector = mesa.DataCollector(
            model_reporters={'Gini': self.compute_gini},
            agent_reporters={
                'Wealth': "wealth", 
                'Steps_not_given': 'steps_not_given'
            }
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def compute_gini(model):
        agent_wealths = [agent.wealth for agent in model.schedule.agents]
        x = sorted(agent_wealths)
        N = model.n_agents
        B = sum(xi * (N-i) for i, xi in enumerate(x)) / (N * sum(x))
        return 1 + (1/N) - 2*B


def plot_wealth_distribution(df, n_agents):
    '''
        Plots frequency of wealth amounts for specified amount of agents in model.
    '''
    df_filtered = df[(df['n_agents'] == n_agents) & (df['Step'] == df['Step'].max())]
    df_filtered = df_filtered.loc[:, ['AgentID', 'iteration', 'Wealth']]
    snshist = sns.histplot(df_filtered['Wealth'], discrete=True)
    snshist.set(title='Wealth distribution', xlabel='Wealth', ylabel='Number of agents')
    plt.show()


def plot_agents_wealth(df, n_agents, iteration, agent_ids):
    '''
        Nefunguje, sekne se
        Plots specified agents wealth over time for specific iteration of simulation
    '''
    df_filtered = df[(df['n_agents'] == n_agents) & (df['iteration'] == iteration)]
    df_filtered = df_filtered.loc[:, ['AgentID', 'Step', 'Wealth']]
    df_filtered = df_filtered[df_filtered['AgentID'].isin(agent_ids)]
    lplot = sns.lineplot(data=df, x='Step', y='Wealth', hue='AgentID')
    lplot.set(title=f'Wealth of agents: {agent_ids} over time.')
    plt.show()


def plot_average_wealth(df, n_agents):
    '''
        Plots average wealth in each step.
    '''
    df_filtered = df[df['n_agents'] == n_agents]
    df_filtered = df_filtered.loc[:, ['AgentID', 'iteration', 'Step', 'Wealth']]
    lplot = sns.lineplot(data=df, x='Step', y='Wealth', errorbar=('ci', 95))
    lplot.set(title='Average wealth over time.')
    plt.show()


def plot_agents_steps_not_given(model, agents_ids):
    df = model.datacollector.get_agent_vars_dataframe()
    df = df[df.index.get_level_values('AgentID').isin(agents_ids)]
    lplot = sns.lineplot(data=df, x='Step', y='Steps_not_given', hue='AgentID')
    lplot.set(title=f'Steps not given of agents: {agents_ids} over time.')
    plt.show()


def plot_average_steps_not_given(df, n_agents):
    '''
        Plots average steps that agent does take without transaction in each step.
    '''
    df_filtered = df[df['n_agents'] == n_agents]
    df_filtered = df_filtered.loc[:, ['AgentID', 'iteration', 'Step', 'Steps_not_given']]
    lplot = sns.lineplot(data=df, x='Step', y='Steps_not_given', errorbar=('ci', 95))
    lplot.set(title=f'Average steps not given over time.')
    plt.show()


# def plot_agent_distribution(df, n_agents, iteration):
#     df_filtered = df[(df['n_agents'] == n_agents) & (df['Step'] == df['Step'].max()) & df['iteration'] == iteration)]
#     df_filtered = df_filtered.loc[:, ['AgentID', 'iteration', 'Wealth']].sort_values(by=['AgentID', 'iteration'])
#     agent_counts = np.zeros((model.grid.width, model.grid.height))
#     for cell_content, (x, y) in model.grid.coord_iter():
#         num_of_agents = len(cell_content)
#         agent_counts[x][y] = num_of_agents
#     hmap = sns.heatmap(agent_counts, cmap='viridis', annot=True, cbar=False, square=True)
#     hmap.figure.set_size_inches(4, 4)
#     hmap.set(title='Number of agents on each cell of the grid.')
#     plt.show()


def plot_gini_vs_step(df, n_agents):
    '''
        For specified number of agents take gini coeff of whatever agent and show 
            progress over steps.
        Output: Plot central tendency and confidence interval that aggregates over 
            multiple y values at each value of x
    '''
    df_filtered = df[df['AgentID'] == 0]
    df_filtered = df_filtered[df_filtered['n_agents'].isin(n_agents)]
    lplot = sns.lineplot(data=df_filtered, x='Step', y='Gini', hue='n_agents')
    lplot.set(
        title=f'Gini coefficient over time (N-agents: {n_agents})', 
        xlabel='Step',
        ylabel='Gini coefficient'
    )
    plt.show()


def plot_gini_vs_nagents(df, err_bars=True):
    #gini coeff is same for every agent so it doesn't matter which ID we choose
    df_filtered = df[(df['AgentID'] == 0) & (df['Step'] == df['Step'].max())]
    df_filtered[['iteration', 'n_agents', 'Gini']].reset_index(drop=True).head()
    if err_bars:
        snsplot = sns.pointplot(data=df_filtered, x='n_agents', y='Gini', linestyle='none')
    else:
        snsplot = sns.scatterplot(data=df_filtered, x='n_agents', y='Gini')
    snsplot.figure.set_size_inches(8, 4)
    snsplot.set(xlabel='Number of agents', ylabel='Gini coefficient', title='Gini coefficient vs. number of agents')
    plt.show()


def main():
    params = {
        'n_agents': [5, 10, 20, 50, 100],
        'space_width': 10,
        'space_height': 10
    }

    results = mesa.batch_run(
        model_cls=BoltzmannWealthModel,
        parameters=params,
        iterations=7,
        max_steps=100,
        data_collection_period=1,
        display_progress=True,
        number_processes=1
    )
    df = pd.DataFrame(results)

    #plot_agents_wealth(df, n_agents=100, iteration=0, agent_ids=[2,6,10])
    plot_average_wealth(df, n_agents=100)
    plot_wealth_distribution(df, n_agents=100)
    plot_gini_vs_step(df, n_agents=[5, 10, 20, 50, 100])
    #plot_agents_steps_not_given(df, agents_ids=[2, 6, 10])
    plot_average_steps_not_given(df, n_agents=100)
    #plot_agent_distribution(df)
    plot_gini_vs_nagents(df)

    df.to_csv('data.csv')


if __name__ == '__main__':
    main()