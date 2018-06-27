# -*- coding: utf-8 -*-
from scrapy.spiders import Spider
# import re
from termcolor import colored

class cnnSpider(Spider):
    name = 'cnnSpider'
    allowed_domains = ['edition.cnn.com']
    start_urls = [
        'http://edition.cnn.com/regions/',
        'http://edition.cnn.com/africa'
        ]

    def parse(self, response):
        for link in response.xpath('//article/div/div[@class="cd__content"]/h3/a/@href').extract():
            if(link.endswith('index.html')):
                full_link = 'https://www.edition.cnn.com'+link
                print(colored(full_link, 'red'))
                yield response.follow(full_link, self.get_article)

    def get_article(self, response):
        title = response.css('h1.pg-headline::text').extract_first()
        print(colored(title, 'green'))
        contents = response.xpath('//article/div[@class="l-container"]/div[@class="pg-rail-tall__wrapper"]/div[@class="pg-side-of-rail pg-rail-tall__side"]/div[@class="pg-rail-tall__body"]/section[@id="body-text"]/div/div[@class="zn-body__read-all"]/div[@class="zn-body__paragraph"]/text()').extract()
        content =  " ".join(str(x) for x in contents)
        # content = re.sub('[^a-zA-Z0-9-_*.]', ' ', content)
        if(title is not None and content != ""):
            yield {
                'title': title,
                'content': content
            }

            
