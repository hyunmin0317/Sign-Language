from time import sleep

from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

def login(id, password):
    URL = "https://ecampus.smu.ac.kr/login.php"

    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(URL)
    driver.find_element_by_id('input-username').send_keys(id)
    driver.find_element_by_id('input-password').send_keys(password)
    driver.find_element_by_name('loginbutton').click()
    return driver.page_source

if __name__ == '__main__':
    url = login("201911019", "1q2w3e4r!!!")
    soup = BeautifulSoup(url, 'html.parser')
    print(soup)