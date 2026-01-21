
from bs4 import BeautifulSoup
import csv
import requests

url = "https://www.passiton.com/inspirational-quotes"

r = requests.get(url)

# soup = BeautifulSoup(R.content,'html5lib')
soup = BeautifulSoup(r.content, 'html5lib')

qoutes = []

table = soup.find("div",attrs={"id":"all_items"})

for row in table.find_all('div',attrs={'class':'text-center mb-8'}):
    qoute = {}
    qoute["Theme"] = row.h5.text
    qoute["url"] = row.a["href"]
    qoute["image"] = row.img["src"]
    qoute["lines"] = row.img["alt"].split[" "][0]
    qoutes.append(qoute)

filename = "web-scraping/inpirationalquotesmethod1.csv"
with open(filename, "w" ,newline='') as f:
    w = csv.DictWriter(f,["Theme","url","image","lines"])
    w.writeheader()
    for qoute in qoutes:
        w.writerow(qoute)






