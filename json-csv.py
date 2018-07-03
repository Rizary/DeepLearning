# coding:utf-8
# /bin/python

import json
import csv

output = open("file.tsv", 'wb')
fileS = csv.writer(output, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
fileS.writerow(['title', 'content'])

with open('file.json') as f:
    for baris in f:
        baris = json.loads(baris)
        fileS.writerow([row['title'], (row['content']).encode('utf-8')])
output.close()