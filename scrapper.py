
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import requests
import pandas as pd 

url='https://en.wikipedia.org/wiki/Economy_of_Finland'
page=requests.get(url)
soup = BeautifulSoup(page.content,'html')

table_column=soup.find('table',class_='wikitable sortable').findAll('th')
table_column_header=[titles.text.strip() for titles in table_column]
df=pd.DataFrame(columns=table_column_header)

table_column_data=soup.find('table',class_='wikitable sortable').findAll('tr')
print(df)
for row in table_column_data: 
    table_row=row.find_all('td')
    table_row_data=[titles.text.strip() for titles in table_row]
    print(table_row_data)
    
#for row in table_column_data: 
 #   filter=row.findAll('td')
    

















