import requests
import pandas as pd
from bs4 import BeautifulSoup

def extract_text_from_url(url,file_id):
    response = requests.get(url)

    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    # text = soup.get_text()
    # print(text)

    title = soup.title.string if soup.title else "N/A"

    paragraphs = soup.find_all('p')
    # for p in paragraphs:
    #     print(p.get_text())

    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()

    # clean_text = soup.get_text(strip=True)
    # print(clean_text)

    clean_text = ' '.join([p.get_text(strip=True) for p in paragraphs])

    with open(file_id, "w", encoding="utf-8") as file:
        file.write("Title: " + title + "\n" + "Article: " + clean_text)


# url = "https://insights.blackcoffer.com/ai-and-ml-based-youtube-analytics-and-content-creation-tool-for-optimizing-subscriber-engagement-and-content-strategy/"  
# extract_text_from_url(url,"./textFolder/hello.txt")

df = pd.read_excel('Input.xlsx')
# print(df.head())

id = df['URL_ID']
link = df['URL']

print(id[4])
