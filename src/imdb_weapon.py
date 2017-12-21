from bs4 import BeautifulSoup
import urllib.request
import requests
import re
import os


default_url = 'http://www.imfdb.org'

def download_images(movie_url):
    print(movie_url)

    movie_name = movie_url.replace(default_url + '/wiki/', '')
    movie_path = "movie_images/" + movie_name
    os.mkdir(movie_path)

    web_content = requests.get(
        movie_url
    ).content.decode('UTF-8')
    soup_data = BeautifulSoup(web_content, 'html.parser')

    image_links = []
    for image in soup_data.find_all('img'):
        re_found = re.search(r'(\/images\/.*\.jpg)', image.get('src'))
        if (re_found):
            link = re_found.group(0)
            if (link.find('poster') == -1 and link.find('SPOILERS') == -1):
                image_links.append(link)

    for link in image_links:
        #print(default_url + link)

        try:
            urllib.request.urlretrieve(
                default_url + link,
                movie_path + '/' + link[link.rfind('/') + 1:]
            )
        except Exception as err:
            print("ERR: " + str(err))


movie_content = requests.get(
    'http://www.imfdb.org/wiki/Category:Movie'
).content.decode('UTF-8')
movie_content = movie_content[
    movie_content.find('Pages in category \"Movie\"'):
    movie_content.find('<div class="printfooter">')
]
soup_data = BeautifulSoup(movie_content, 'html.parser')

li_list = [li.get('href') for li in soup_data.find_all('a')]
next_link = li_list[-1]

#print('\n'.join(li_list))

for link in li_list[:-1][1:]:
    download_images(default_url + link)
