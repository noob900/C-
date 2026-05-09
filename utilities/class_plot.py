import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('economy_of_finland.csv')

class plot_data(pd.DataFrame):
    df=pd.read_csv('economy_of_finland.csv')
    gdp_values = df['GDP'].to_list()
    Unemployment=df['Unemployment'].to_list()
    years = df['Year'].to_list()

    def __init__(self,df):
        self.df=df

    def plot_data(self):
        plt.plot(self.df['Year'],self.df['Value'])
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.title('Economy of Finland')
        plt.show()


    # Create an instance of plot_data
    plotter = plot_data(df)

    # Call the plot_data method to plot the data
    plotter.plot_data()