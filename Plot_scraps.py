import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('economy_of_finland.csv')  

gdp_values = df['GDP'].to_list()
Unemployment=df['Unemployment'].to_list()
years = df['Year'].to_list()

fig, ax1 = plt.subplots()


ax1.set_xlabel('Year')
ax1.set_ylabel('GDP (in Bil. Euro)')
ax1.plot(years, gdp_values, color='red')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.set_ylabel('Unemployment')
ax2.plot(years, Unemployment, color='green')
ax2.tick_params(axis='y')

plt.title('GDP and Unemployment of Finland over the Years')
plt.grid(True)
plt.show()
