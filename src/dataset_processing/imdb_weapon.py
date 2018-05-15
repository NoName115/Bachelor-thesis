# Script for downloading images from imfdb
#
# Author: Robert Kolcun, FIT
# <xkolcu00@stud.fit.vutbr.cz>

from bs4 import BeautifulSoup
import urllib.request
import requests
import re
import os


default_url = 'http://www.imfdb.org'
images_folder = 'move_images_2/'

def download_images(movie_url):
    def get_image_link(image_url):
        image_web_content = requests.get(
            default_url + image_url
        ).content.decode('UTF-8')

        image_web_content = image_web_content[
            image_web_content.find('<div id="contentSub"></div>'):
            image_web_content.find('</div><h2 id="filehistory">File history</h2>')
        ]
        content_soup = BeautifulSoup(image_web_content, 'html.parser')

        for aa in content_soup.find_all('a'):
            href = aa.get('href')
            if (href and href.find('images') != -1 and href.find('thumb') == -1):
                return href

        return None

    print("-------- MOVIE ----------")
    print(movie_url)

    movie_name = movie_url.replace(default_url + '/wiki/', '')
    movie_path = images_folder + movie_name.replace('/', '_')
    if (os.path.exists(movie_path)):
        return
    
    os.mkdir(movie_path)

    web_content = requests.get(
        movie_url
    ).content.decode('UTF-8')
    soup_data = BeautifulSoup(web_content, 'html.parser')

    href_list = []
    for aa in soup_data.find_all('a'):
        href = aa.get('href')
        if (href and href.find('File:') != -1):
            href_list.append(href)

    # Filter every second element
    href_list = href_list[::2]

    for href in href_list:
        image_link = get_image_link(href)
        print(image_link)
        try:
            urllib.request.urlretrieve(
                default_url + image_link,
                movie_path + '/' + image_link[image_link.rfind('/') + 1:]
            )
        except Exception as err:
            print("ERR: " + str(err))

# Downloaded films link
# 'http://www.imfdb.org/wiki/Category:Movie'

movie_content = requests.get(
    'http://www.imfdb.org/index.php?title=Category:Movie&pagefrom=Corsican+File%2C+The+%28L%27enqu%C3%AAte+Corse%29#mw-pages'
).content.decode('UTF-8')
movie_content = movie_content[
    movie_content.find('Pages in category \"Movie\"'):
    movie_content.find('<div class="printfooter">')
]
soup_data = BeautifulSoup(movie_content, 'html.parser')

li_list = [li.get('href') for li in soup_data.find_all('a')]
next_link = li_list[-1]

#print('\n'.join(li_list))
if (not os.path.exists(images_folder)):
    os.mkdir(images_folder)

for link in li_list[:-1][1:]:
    download_images(default_url + link)
