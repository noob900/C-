import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self, data):
        self.df = pd.read_csv(data)
        self.unemployment = self.df['Unemployment'].to_list()
        self.years = self.df['Year'].to_list()

        self.years=np.sort(self.years)

        self.fig, self.ax1 = plt.subplots()

        # Plot data
        self.ax1.plot(self.years, self.unemployment, 'o-')

        # Set labels and ticks
        self.ax1.set_ylabel('Unemployment rate (%)')
        self.ax1.set_xlabel('Year')

        # Set y-axis ticks from 0 to 20, ensuring they are in ascending order
        yticks = np.arange(0, len(self.unemployment)+1, 4)
        self.ax1.set_yticks(yticks)

        # Set x-axis ticks every 10 years
        self.ax1.set_xticks(np.arange(1980, 2031, 10))

        plt.title('Economy of Finland over the Years')
        plt.show()

# Usage
if __name__ == "__main__":
    plot = Plot('economy_of_finland.csv')
