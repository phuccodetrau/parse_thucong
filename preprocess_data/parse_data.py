import fitz
from PIL import Image
import io
import base64
from io import BytesIO
import requests
import time
import re

class ParseHandler():
    _instance = None
    prompt_system = """You are an AI assistant that processes images containing legal content. Your task is to transcribe the content from the image into markdown format with the following rules:
        
                        1. Use `#` for titles if you see the name of a Law or Decree.
                        2. Use `##` if you see the word "Chương" or "Mẫu số" at the beginning of a sentence.
                        3. Use `###` if you see the word "Mục" at the beginning of a sentence.
                        4. Use `####` if you see the word "Điều" at the beginning of a sentence.
                        5. Do not use any '#' for a), b), c),... etc or 1), 2), 3),...etc
                        6. If you encounter a equation, notice the following rules:
                            - The equation can be on multiple lines, detect all the lines of the equation.
                             - Convert the method in the image into a single latex block with all expressions inside a pair of dollar signs ($$). Try to get both legs before and after the equal sign(=) (i.e. $$ \text{Gross bonus amount} = \text{Bonus} \times \left(100 - \frac{\text{Date of absent in a year}}{22 \times 12}\right) $$)
                        7. Only respond with the markdown content, without any additional explanation or description.
                        8. Don't annotate ```markdown\n  in the response
                    """
    detect_table_prompt = """You are an expert in table dectection. If the input image have any tables,focus on each table and response the markdown of each table with the index of each table in the begin of table (i.e. Table 1, Table 2, ...etc). Try to keep all the text of each table at the right cell in the image and do not annotate ```markdown. If the image don't have any tables, response "0" """

    @staticmethod
    def get_instance(api_key):
        """ Static access method. """
        if not ParseHandler._instance:
            ParseHandler._instance = ParseHandler(api_key)
        return ParseHandler._instance
    
    def __init__(self, api_key) -> None:
        self.api_key = api_key

    def pdf_to_images(self, file_stream, pdf_path, zoom_x=2.0, zoom_y=2.0):
        pdf_document = fitz.open(stream=file_stream, filetype="pdf")
        images = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            mat = fitz.Matrix(zoom_x, zoom_y)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)
        image_base64s = []

        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            image_base64s.append(base64.b64encode(image_bytes).decode("utf-8"))
        return image_base64s
    
    def detect_table(self, image_base64):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": self.detect_table_prompt
                    },
                ]
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": """Detect table in this image"""
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": 4096
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()['choices'][0]['message']['content']
    
    def concatenate_pages(self, pages, file_name):
        pdf_content = ""
        for page in pages:
            i = page["page"]
            page_content = f"* Trang {i}\n\n" + page["content"] + f"\n\n"
            pdf_content += page_content
        pdf_info = {"content": pdf_content, "reference": pages[0]["reference"], "src": file_name}
        return pdf_info
    
    def parse_pdf(self, image_base64s, file_name):
        pages = []
        tables = []
        law_name = ""
        for i in range(0, len(image_base64s)):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                    "role": "system",
                    "content": [
                        {
                        "type": "text",
                        "text": self.prompt_system
                        },
                    ]
                    },
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": "Transcribe the content from this image into markdown format"
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64s[i]}"
                        }
                        }
                    ]
                    }
                ],
                "max_tokens": 4096
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            page_content = response.json()['choices'][0]['message']['content']
            if i == 0:
                pattern = [r'(BỘ LUẬT\s*\n)(.*)', r'(LUẬT\s*\n)(.*)', r'(NGHỊ ĐỊNH\s*\n)(.*)', r'(QUY ĐỊNH\s*.*\n)(.*)']
                text = ""
                for p in pattern:
                    text = re.findall(p, page_content)
                    if len(text) > 0:
                        law_name = text[0][0].strip() + " " + text[0][1].strip()
                        break

            pages.append({"page": i+1, "content": page_content, "reference": law_name})
            num_of_tables = self.detect_table(image_base64s[i])
            if num_of_tables != "0":
                tables.append({"page": i+1, "content": num_of_tables, "reference": law_name, "src": file_name})            
            time.sleep(2)
        
        pdf_info = self.concatenate_pages(pages=pages, file_name=file_name)
        return pdf_info, tables
