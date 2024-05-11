from bs4 import BeautifulSoup
import requests
url='https://en.wikipedia.org/wiki/Economy_of_Finland'
page=requests.get(url)
soup = BeautifulSoup(page.content,'html')
table=soup.find('table',class_='wikitable sortable').findAll('th')

header=[titles.text.strip() for titles in table]

print(header)








