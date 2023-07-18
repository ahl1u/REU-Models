#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 19:49:05 2021
# NOTE: Please remove _normal from user image url!!!!!!!!!!!!!!!!!!!!!!!!!

@author: sunli
Modified by: Zidian
"""
# sign up on face++ webside first to get the key and the secret

import csv
import requests
from json import JSONDecoder
from PIL import Image
import io

http_url='https://api-us.faceplusplus.com/facepp/v3/detect'

key ="H37UklfKuzHXlwKylF0_Xq4MPCUG5acV"  # input key sign by face++
secret ="T-sglqURhoQMD2sci3YM7OHlRE0xmD38"  # input secret sign by face++

data = {"api_key":key, "api_secret": secret,"return_attributes": "gender,age,ethnicity,emotion,smiling"}

def resize_image(image, size=(224, 224)):
    """Resize an image to the specified size"""
    return image.resize(size, Image.ANTIALIAS)

def face(line):
    url = line[1]
    r = requests.get(url)
    image = Image.open(io.BytesIO(r.content))
    image = resize_image(image)
    image.save('resized_profile.jpg')
    files = {"image_file": open('resized_profile.jpg', "rb")}
    response = requests.post(http_url, data=data, files=files)
    req_con = response.content.decode('utf-8')
    face_info = JSONDecoder().decode(req_con)
    return face_info 

with open ("plswrkP.csv",'r',errors="ignore") as src:
    reader = csv.reader(src)
    next(reader)  # Skip the headers
    total_count=0
    count1=0

    for line in reader:
        total_count+=1
        try:
            faceinfo=face(line)
            if 'face_num' in faceinfo and faceinfo['face_num'] == 1:  
                count1+=1
                with open("pfpresults2.csv", 'a', newline='', encoding="utf-8") as output_file:
                    writer=csv.writer(output_file)
                    gender = faceinfo['faces'][0]['attributes']['gender']['value']
                    age = faceinfo['faces'][0]['attributes']['age']['value']
                    writer.writerow((line[0], line[1], age, gender))
                    print('total count: ' + str(total_count))
                    print('count1: ' + str(count1))
        except Exception as e:
            print(f"Error occurred on line {total_count}: {e}")
            continue
