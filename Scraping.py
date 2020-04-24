import pandas as pd
from datetime import datetime
from datetime import timedelta
import requests as req
import re
import numpy as np
from bs4 import BeautifulSoup
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))

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
