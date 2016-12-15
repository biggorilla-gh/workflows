# This script crawls a couple of wiki urls, extracts the titles and
# the first paragraphs and stores them in a json file.
import urllib2
import json
from bs4 import BeautifulSoup

data = []
header = {'User-Agent': 'Mozilla/5.0'} #Needed to prevent 403 error on Wikipedia
wiki_urls = [
	'https://en.wikipedia.org/wiki/Adobe',
	'https://en.wikipedia.org/wiki/Brick',
	'https://en.wikipedia.org/wiki/Concrete',
	'https://en.wikipedia.org/wiki/Trunk_(botany)',
	'https://en.wikipedia.org/wiki/Metal',
	'https://en.wikipedia.org/wiki/Stone_(disambiguation)',
	'https://en.wikipedia.org/wiki/Rock_(geology)',
	'https://en.wikipedia.org/wiki/Straw',
	'https://en.wikipedia.org/wiki/Wood'
]

for wiki in wiki_urls:
	feature_dict = {}
	req = urllib2.Request(wiki,headers=header)
	page = urllib2.urlopen(req)

	#Parse the html in the 'page' variable, and store it in Beautiful Soup format
	soup = BeautifulSoup(page, 'html.parser')

	feature_dict["description"] = soup.p.get_text()
	feature_dict["title"] = soup.h1.get_text()
	data.append(feature_dict)


with open('wiki_home_construction_features.json', 'w') as jsonData:
    json.dump(data, jsonData)
