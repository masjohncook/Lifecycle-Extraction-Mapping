import pytesseract  ## Convert pdf into page
from pdf2image import convert_from_path
import glob
import asyncio
from pyppeteer import launch
from spire.pdf.common import *  ## Extract the image from the pdf
from spire.pdf import *
import nltk.data ## Find the sentence
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from rapidfuzz import fuzz, utils
import trafilatura  ## Extract the main content
import collections
from llama_index.llms import Ollama
from pathlib import Path
import re
import requests
import json
import time
tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

async def generate_pdf(url, pdf_path):
    browser = await launch()
    page = await browser.newPage()

    await page.goto(url)

    await page.pdf({'path': pdf_path, 'format': 'A4'})

    await browser.close()
class Process_Image:
    def __init__(self):
        #nltk.download('stopwords')
        self.pdfs = glob.glob(r"html_file/*.pdf")
        self.texts = glob.glob(r"html_file/*.txt")
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.link = "https://www.hackingarticles.in/escalate_linux-vulnhub-walkthrough-part-1/"
        self.llm = Ollama(model="vicuna:13b")  # Run locally

    def llm_inference(self, content):
        response = self.llm.complete(content)

        response = str(response)
        # Split the content by lines
        lines = response.split('\n')

        ## Split the response into sentences
        # Extract sentences
        sentences = []
        for line in lines:
            # Check if line starts with a number (indicating a new sentence)
            if line.strip() and line[0].isdigit():
                # Extract the sentence without the leading number
                sentence = ' '.join(line.split()[1:])
                sentences.append(sentence)
        return sentences

    def image_inference(self, content):
        response = self.llm.complete(content)

        response = str(response)

        return  response
    def pdf_to_text_pages(self): # Independent (set the path)
        ## Convert the pdf into text page by page
        for pdf_path in self.pdfs:  ## loop the pages in one pdf
            pages = convert_from_path(pdf_path, 500)

            for pageNum,imgBlob in enumerate(pages):
                text = pytesseract.image_to_string(imgBlob,lang='eng') #convert the image to string

                with open(f'{pdf_path[:-4]}_page{pageNum}.txt', 'w',encoding="utf-8") as the_file:

                    the_file.write(text)

    def is_valid_url(self, url):
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            # A valid URL typically returns a status code less than 400
            return response.status_code < 400
        except requests.RequestException:
            # If any exception occurred while making the request, the URL is invalid
            return False
    def extract_iamge_from_page(self, path):  # Independent (Set the path)
        ## Extract the image from the pdf page by page

        # Create a PdfDocument object
        doc = PdfDocument()

        # Load a PDF document
        doc.LoadFromFile('{path}/report.pdf'.format(path=path))

        # Get a specific page
        page = doc.Pages[0]

        # Loop through the pages in the document
        for l in range(doc.Pages.Count):
            index = 0
            page = doc.Pages.get_Item(l)
            # Extract images from a specific page
            for image in page.ExtractImages():
                imageFileName = '{path}/Image_page{p}_{d}.png'.format(d=index, p=l, path=path)
                index += 1
                image.Save(imageFileName, ImageFormat.get_Png())
        doc.Close()

    def image_to_text(self, image_file): # extract the name from file

        ## Separate the paragraph into sentences line by line
        a = str(pytesseract.image_to_string(image_file))  ## convert the image into the string
        combine = self.tokenizer.tokenize(a)
        if a == "" or len(a) < 10:
            return ""
        return a


    def text_similarity(self,text1, text2):
        similarity = fuzz.QRatio(text1,text2, processor=utils.default_process)
        return similarity

    def get_text_pdf(self):
        # Count the text in the file
        texts = self.texts

        return len(texts)
    def get_main_content_url (self):
        downloaded = trafilatura.fetch_url(self.link)
        text = trafilatura.extract(downloaded)
        combine = self.tokenizer.tokenize(text)

        return combine ## straight return the whole text

    def extract_files_by_page(self, directory, page_number):
        matching_files = []

        # List all files in the directory
        files = os.listdir(directory)

        # Define the regex pattern to match file names
        pattern = re.compile(fr'Image_page{page_number}_(\d+)\.png')

        for file_name in files:
            file_path = os.path.join(directory, file_name)

            # Check if the file is a regular file and ends with '.png'
            if os.path.isfile(file_path) and file_name.endswith('.png'):
                # Match the file name with the specified pattern
                match = pattern.match(file_name)
                if match:
                    matching_files.append(file_path)

        return matching_files

    def compare_funtion(self, text, main_content, image_content):
        text = str(text)
        for i in main_content:
            if self.text_similarity(text,i) > 50:
                return text
        for x in range (len(image_content)):
            for e in image_content[x]:
                if self.text_similarity(text,e) > 60:
                    return "img_{i}".format(i=x)
        return None

    def page_info(self, page_number, path):
        print("\n\n\n\nThis is page ", page_number, "\n")
        small_arrange = []
        directory_to_search = path
        file_text_page = "{path}/report_page{d}.txt".format(d=page_number,path=path)
        matching = self.extract_files_by_page(directory_to_search,page_number)
        img_dict ={}
        prompt = """ Separate the text in the content into sentences and list them with numbers 
        Content = [{content}]
                """
        img_count_flag = True
        counter = 0

        while(img_count_flag):
            with open(file_text_page, encoding="utf-8") as f:
                content = f.read()  ## Content of respective page
                new_content = prompt.format(content=content)
                file_text_sentence = self.llm_inference(new_content)

                print("Text Done")
                #time.sleep(2)

            if len(matching) == 0:
                print("No Image")
                return file_text_sentence

            for b in range(len(matching)):
                image_content = self.image_to_text(matching[b])
                if image_content == "" :
                    img_dict[b] = ""
                else:
                    new_image_content = prompt.format(content=image_content)
                    #time.sleep(2)
                    img_dict[b] = self.llm_inference(new_image_content)
            print("Image Done")
            main_content = self.get_main_content_url()
            first_part, second_part = main_content[:len(main_content) // 2], main_content[len(main_content) // 2:]
            new_main_content = prompt.format(content=first_part)
            main_content_sentence = self.llm_inference(new_main_content)
            print("First Main Done")
            time.sleep(2)
            main_content_sentences = main_content_sentence
            new_main_content = prompt.format(content=second_part)
            main_content_sentence = self.llm_inference(new_main_content)
            #time.sleep(2)
            main_content_sentences += main_content_sentence
            print("Second Main Done")


            A_series = pd.Series(file_text_sentence)
            # Use apply with extra arguments
            result = A_series.apply(self.compare_funtion,args=(main_content_sentences, img_dict))
            # Convert result to list
            result_list = result.tolist()

            # Removing None values from the list
            result_list = [item for item in result_list if item is not None]
            ### Remove duplicate img
            merged_list = []
            seen_imgs = set()

            for items in result_list:
                if items.startswith('img_'):
                    if items not in seen_imgs:
                        merged_list.append(items)
                        seen_imgs.add(items)
                else:
                    merged_list.append(items)

            print(merged_list)
            # Using a set to count unique 'img_x' entries
            images = set(item for item in merged_list if item.startswith('img_'))
            img_count = len(images)

            if img_count == len(matching):
                img_count_flag = False
            else:
                print("Expected ", len(matching), "But only: ", img_count)
                counter += 1
                print("Counter: ", counter)

            if counter == 3:
                print("Seems like the picture is not captured by OCR, add manually....")
                if len(matching) == 1:
                    for i in range(len(matching)):
                        merged_list.append("img_{i}".format(i=i))
                else:
                    # Extracting and counting the number of unique 'img_x' entries
                    img_numbers = set()
                    for item in merged_list:
                        if item.startswith('img_'):
                            # Extract the number part from 'img_x' and add it to the set
                            num_part = item.split('_')[1]
                            img_numbers.add(num_part)
                    for m in range(len(matching)): ## Some image have captured, append the non-captured imaged to the last
                        if str(m) in img_numbers:
                            pass ## Do nothing
                        else:
                            merged_list.append("img_{i}".format(i=m))
                print ("The new list: ", merged_list)
                img_count_flag = False

        prompt = """ Summarize the content in one paragraph and understandable by non-IT people.
        Content = [{content}]
                """

        # Construct the content of image for replacement
        matching_dict ={}
        for j in range(len(matching)):
            image_content = self.image_to_text(matching[j])
            if image_content =="":
                matching_dict["img_{i}".format(i=j)] = ""
            else:
                new_image_content = prompt.format(content=image_content)
                interpret_result = self.image_inference(new_image_content)
                time.sleep(2)
                matching_dict["img_{i}".format(i=j)] = interpret_result

        # Replace 'img_x' with specified content
        replaced_list = []
        for item in merged_list:
            if item in matching_dict:
                # Split the text into sentences and add them to the list
                sentences = matching_dict[item].split(". ")
                replaced_list.extend([sentence.strip() for sentence in sentences if sentence])
            else:
                replaced_list.append(item)
        print(replaced_list)
        return replaced_list



    def one_shot_power(self, url, number):
        print("\nStart, ", number)
        # Check the URL
        if self.is_valid_url(url):
            pass
        else:
            return (url, " is not valid")

        # Create independent folder for each url
        folder_name = "report/url_{i}".format(i=number)
        try:
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' has been created.")
        except FileExistsError:
            print(f"Folder '{folder_name}' already exists.")

        #asyncio.get_event_loop().run_until_complete(generate_pdf(url, folder_name+"/report.pdf"))  ## Convert the url to pdf
        self.pdfs = glob.glob(r"{file}/*.pdf".format(file=folder_name))
        self.texts = glob.glob(r"{file}/*.txt".format(file=folder_name))
        self.link = url

        # convert the pdf into text
        self.pdf_to_text_pages()

        # extract the image from pdf
        self.extract_iamge_from_page (folder_name)

        total_text_file = self.get_text_pdf()
        print(total_text_file)

        for t in range(total_text_file):
            print("Hi")
            info = self.page_info(t, folder_name)


            for sentence in info:
                with open("{path}/result.txt".format(path=folder_name), 'a', encoding="utf-8") as file:
                    file.writelines(sentence+ "\n")
            with open("{path}/reference.txt".format(path=folder_name), 'a', encoding="utf-8") as file:
                file.writelines(info)
                file.writelines("\n")

        print ("Complete, " ,number,"\n")

if __name__=='__main__':
    with open('all_report_link.txt',  encoding="utf-8") as file:
        string_data = file.read()

    ignore = [0, 1, 2,3, 4,5, 6, 7, 8, 9, 10, 11, 12 ,13 ,14 ,15, 16, 17, 18, 19, 20, 21, 22, 23,24,25, 26, 27, 28, 29,
              30, 31, 32, 33,34,35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,46,  48, 49,50,51,52,53,54,55,56,57,58,59,60,
              61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,
    81,82, 89]
    url_list = string_data.split('\n')
    worker  = Process_Image()
    for z in range(len(url_list)):
        if z in ignore:
            continue
        worker.one_shot_power(url_list[z], z)
        #time.sleep(10)
    #worker.one_shot_power(url_list[2],2)



    pass
    #worker = Process_Image()
    # worker.one_shot_power()
    #worker.page_info(8)
    # Run the function
    #asyncio.get_event_loop().run_until_complete(generate_pdf(link, 'html_file/example1.pdf'))  ## Convert the url to pdf
    # hold_dict={}
    # for i in range (2):
    #     info = worker.page_info(i)
    #     hold_dict["page_{i}".format(i=i)] = info
    #     print (info)
    # print (hold_dict)
