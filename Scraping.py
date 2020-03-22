import pandas as pd
from datetime import datetime
from datetime import timedelta
import requests as req
import re
import numpy as np
from bs4 import BeautifulSoup

page=req.get('https://www.nytimes.com/search?dropmab=true&query=coronavirus&sort=newest')
soup=BeautifulSoup(page.text,'html.parser')
#print(soup.prettify())
#list=soup.find(class_='data-testid').prettify()
soup.find_all('a')
list=soup.ol.find_all('li')
#print(len(link.contents))
for link in list:
    #print(len(link.contents))

    print(link.contents[0].a.h4)
    print(link.contents[0].a)

#    soup.ol.
#    print(link.contents)
#print(list)
#for link in soup.find_all('a'):
    #print(link.get('href'))
#list=soup.find(class_='table').prettify()
#list_data=pd.read_html(list)[0]
#list_data.columns=['Rank','Classification','Date','Region','Start_Time','Maximum_Time','End_Time','Movie']
#list_data.head(10)
