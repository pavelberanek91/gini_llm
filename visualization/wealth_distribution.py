import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_wealth_distribution(df, n_agents):
    '''
        Plots frequency of wealth amounts for specified amount of agents in model.
    '''
    df_filtered = df[(df['n_agents'] == n_agents) & (df['Step'] == df['Step'].max())]
    df_filtered = df_filtered.loc[:, ['AgentID', 'iteration', 'Wealth']].sort_values(by=['AgentID', 'iteration'])
    snshist = sns.histplot(df_filtered['Wealth'], discrete=True)
    snshist.set(title='Wealth distribution', xlabel='Wealth', ylabel='Number of agents')
    plt.show()


def main():
    CSV_SOURCE = 'data.csv'
    df = pd.read_csv(CSV_SOURCE)
    plot_wealth_distribution(df, n_agents=50)


if __name__ == '__main__':
    main()
