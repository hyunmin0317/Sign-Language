from time import sleep

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

def login(id, password):
    URL = "https://ecampus.smu.ac.kr/login.php"

    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(URL)
    driver.find_element(By.ID, 'input-username').send_keys(id)
    driver.find_element(By.ID, 'input-password').send_keys(password)
    driver.find_element(By.NAME, 'loginbutton').click()
    return driver.page_source

if __name__ == '__main__':
    url = login("201911019", "1q2w3e4r!!!")
    soup = BeautifulSoup(url, 'html.parser')
    courses = soup.find_all("div", class_="course-title")

    print(" 강좌 개수: "+str(len(courses))+"\n\n 수업 목록")
    for course in courses:
        print(" 강좌: "+course.find("h3").text)
        print(" 교수: "+course.find("p").text)