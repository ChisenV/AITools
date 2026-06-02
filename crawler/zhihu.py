import json
import re
from datetime import datetime
import os

os.environ["PATH"] = r"E:\thirdparty\TensorRT\TensorRT-10.10.0.31\lib;" + os.environ["PATH"]

import AITools
from bs4 import BeautifulSoup


# ========= 规则 =========
TARGET_FLAG = "CV计算机视觉每日开源代码Paper with code速览"

GITHUB_PATTERN = re.compile(r'https?://github\.com/[^\s)]+')
ARXIV_PATTERN  = re.compile(r'https?://arxiv\.org/[^\s)]+')


def extract_articles(html: str):
    soup = BeautifulSoup(html, "html.parser")

    results = []

    # 1️⃣ 找所有容器
    containers = soup.find_all("div", class_="css-9w3zhd")

    for div in containers:
        # 2️⃣ 在当前容器内找标题 a（限制范围，避免污染）
        a = div.select_one("h2 a")
        if not a:
            continue

        href = a.get("href", "").strip()
        title = a.get_text(strip=True)

        if not href or not title:
            continue

        results.append({
            "url": href,
            "title": title
        })

    return results


# ========= 抽取并配对 =========
def extract_pairs(text: str):
    """
    返回：
    [
        {paper_url, code_url}
    ]
    """

    pairs = []

    # 找到所有 arxiv 出现位置
    for match in ARXIV_PATTERN.finditer(text):
        paper_url = match.group()
        paper_pos = match.start()

        # 在附近找 github（向后找更合理）
        window_end = min(paper_pos + 300, len(text))
        window_text = text[paper_pos:window_end]

        githubs = GITHUB_PATTERN.findall(window_text)

        code_url = githubs[0] if githubs else None

        pairs.append({
            "paper_url": paper_url,
            "code_url": code_url
        })

    return pairs


def process_file(input_file: str, output_file: str):
    res_all = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = AITools.JSONParser.loads(line)
            print(data["content_id"], data["created_time"])
            dt = datetime.fromtimestamp(data["created_time"])
            date = dt.strftime("%Y-%m-%d %H:%M:%S")
            papers = extract_pairs(data["content_text"])
            if len(papers) > 0:
                res_all.append({
                    "content_id": data["content_id"],
                    "created_timestamp": data["created_time"],
                    "created_time": date,
                    "papers": papers
                })

    AITools.JSONParser.dump(res_all, output_file, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    process_file(
        r"E:\python_project\MediaCrawler\data\zhihu\jsonl\detail_contents_2026-04-24-01.jsonl",
        ""
    )

    # html = r"C:\Users\WQS\Downloads\(35 封私信 _ 80 条消息) CV每日Paper with code - 知乎.html"
    # with open(html, "r", encoding="utf-8") as f:
    #     content = f.read()
    # articles = extract_articles(content)
    # print(len(articles))
    #
    # for item in articles:
    #     print("\"" + item["url"] + "\",")
    #     # print(item["title"])
    #     # print("------")

