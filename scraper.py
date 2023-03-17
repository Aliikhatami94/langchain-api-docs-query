import requests
from bs4 import BeautifulSoup
import urllib.parse
import html
import re


def extract_local_links(soup, domain, domain_full):
    local_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith(domain) or href.startswith('./') \
                or href.startswith('/') or href.startswith('modules') \
                or href.startswith('use_cases'):
            local_links.append(urllib.parse.urljoin(domain_full, href))
    return local_links


def extract_main_content_text(soup):
    main_content = soup.select('body main')[0]
    main_content_text = main_content.get_text()
    main_content_text = re.sub(r'<[^>]+>', '', main_content_text)
    main_content_text = ' '.join(main_content_text.split())
    main_content_text = html.unescape(main_content_text)
    return main_content_text


def scrape_single_page(url, domain, domain_full):
    res = requests.get(url)
    if res.status_code != 200:
        print(f"{res.status_code} for '{url}'")
        return None
    soup = BeautifulSoup(res.text, 'html.parser')
    local_links = extract_local_links(soup, domain, domain_full)
    main_content_text = extract_main_content_text(soup)
    return {
        "url": url,
        "text": main_content_text
    }, local_links


def scrape_website(url, domain):
    domain_full = domain + "en/latest/"
    res = requests.get(url)
    if res.status_code != 200:
        print(f"{res.status_code} for '{url}'")
        return None
    soup = BeautifulSoup(res.text, 'html.parser')
    local_links = extract_local_links(soup, domain, domain_full)
    data = []
    scraped = set()
    while True:
        if len(local_links) == 0:
            print("Complete")
            break
        url = local_links[0]
        print(url)
        res = scrape_single_page(url, domain, domain_full)
        scraped.add(url)
        if res is not None:
            page_content, new_local_links = res
            data.append(page_content)
            local_links.extend(new_local_links)
            local_links = list(set(local_links))
        local_links = [link for link in local_links if link not in scraped]
    return data
