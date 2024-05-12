from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import requests
import pandas as pd 

url='https://en.wikipedia.org/wiki/Economy_of_Finland'
page=requests.get(url)
soup = BeautifulSoup(page.content,'html')

table_column=soup.find('table',class_='wikitable sortable').findAll('th')
table_column_header=[titles.text.strip() for titles in table_column]
table_column_data=soup.find('table',class_='wikitable sortable').findAll('tr')

df=pd.DataFrame(columns=table_column_header)
for row in table_column_data[1:]: 
       table_row=row.find_all('td')
       array_size=len(table_column_data)
       table_row_data= [titles.text.strip() for titles in table_row]
       df.loc[len(df)]=table_row_data 
  
df.to_csv('economy_of_finland.csv')  
df=pd.read_csv('economy_of_finland.csv')  
df.head()
df.plot(x='Unemployment (in%)',kind='bar')
plt.show()
