import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen

def course():
    URL = 'https://ecampus.smu.ac.kr/'

def main():
    URL = 'https://ecampus.smu.ac.kr/report/ubcompletion/user_progress.php?id=63139'
    html = requests.get(URL)
    print(html.text)
    print(html.url)
    # soup = BeautifulSoup(html.text, 'html.parser')
    # print(soup)

    # response = requests.post(URL).text
    # print(response)
    # soup = BeautifulSoup(response, 'lxml', from_encoding='utf-8')

    urls = []
    data = []

    # print(soup)
    # for href in soup.find("div", class_='result-list').find_all("dt"):
    #     urls.append(BASIC+href.find("a")["href"])
    #
    #
    #
    # for url in urls:
    #     soup = BeautifulSoup(urlopen(url), 'html.parser')
    #     d = []
    #
    #     try:
    #         for content in soup.find("div", class_='file-meta-table-pc').find_all("tr"):
    #             td = (content.find('td').text).replace(u'\xa0', u'').replace(u'\t', u'').replace(u'\n', u'')
    #             d.append(td)
    #
    #         if len(d) == 14:
    #             d.append(url)
    #         else:
    #             url = d[10]
    #             d[10:] = d[11:]
    #             d.append(url)
    #         data.append(d)
    #         print(len(data))
    #     except:
    #         pass
    #
    # df = pd.DataFrame(data)
    # print(df)
    # df.to_csv('public-data.csv', index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    main()