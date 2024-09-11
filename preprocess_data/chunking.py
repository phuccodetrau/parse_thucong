# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chat_models import ChatOpenAI
from llama_index.llms.openai import OpenAI
from llama_index.core import ChatPromptTemplate
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.llms import ChatMessage
import re
import json
DEFAULT_SYSTEM_PDFPARSING_PROMPT= """\
Bạn là người quản lý một nhóm các đoạn văn bản, bảng hoặc biểu đồ, trong đó mỗi đoạn đều đại diện cho một nhóm các câu nói về một chủ đề tương tự nhau.
Bạn cần tạo ra một tóm tắt ngắn gọn trong 1 câu để thông báo cho người xem biết đoạn văn bản, bảng hoặc biểu đó nói về chủ đề gì.

Một tóm tắt tốt sẽ nói lên nội dung của đoạn văn bản, bảng hoặc biểu đồ

Bạn sẽ nhận được một đề xuất mới sẽ được thêm vào đoạn văn bản, bảng hoặc biểu mới. Đoạn văn bản, bảng hoặc biểu đồ mới này cần có một tóm tắt.

Tóm tắt của bạn nên có tính khái quát tương đối. Nếu bạn nhận được một đề xuất về táo, hãy khái quát nó thành thực phẩm.
Hoặc nếu là tháng, hãy khái quát nó thành "ngày và thời gian".

Ví dụ:
Đầu vào: Đề xuất: Greg thích ăn pizza
Đầu ra: Đoạn văn bản này chứa thông tin về các loại thực phẩm mà Greg thích ăn.

Chỉ phản hồi lại bằng tóm tắt của đoạn văn bản mới, không thêm gì khác.
"""

class ChunkHandler():
    _instance = None

    @staticmethod
    def get_instance(api_key:str):
        """ Static access method. """
        if not ChunkHandler._instance:
            ChunkHandler._instance = ChunkHandler(api_key)
        return ChunkHandler._instance
    
    def __init__(self, api_key:str) -> None:
        self.api_key = api_key

    def get_chunk_summary(self, content:str) -> str:
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",DEFAULT_SYSTEM_PDFPARSING_PROMPT,
                ),
                ("user", "Xác định tóm tắt của đoạn văn bản mới mà đề xuất này sẽ được thêm vào:\n{content}"),
            ]
        )
        llm= OpenAI(
            model="gpt-4o-mini",
            system_prompt= DEFAULT_SYSTEM_PDFPARSING_PROMPT,
            api_key=self.api_key,
            temperature=0,
            max_tokens=4096,
            logprobs=None,
            default_headers={},
        )
        new_chunk_summary=llm.predict(prompt=PROMPT, content=content)
        return new_chunk_summary
    
    def create_article(self, pdf_info):
        muc_luc = list()
        pattern = r'(\* Trang \d+)(.*?)(?=\* Trang \d+|$)'
        article_pattern = r"#+\s+Điều\s+\d+[:.]\s+[^\n]+"
        matches = re.findall(pattern, pdf_info["content"], re.DOTALL)
        pages = [match[0] + match[1] for match in matches]
        for i, page in enumerate(pages, 1):
        
            a_matches = re.findall(article_pattern, page)
            for a in a_matches:
                muc_luc.append({"object": a, "page": i})
        

        documents = []
        chapter_pattern = r'(#+\s+Chương\s+[IVXLCDM]+.*?)(?=#+\s+Chương\s+[IVXLCDM]+|$)'


        chapters = re.findall(chapter_pattern, pdf_info["content"], re.S)

        if len(chapters) == 0:
            article_pattern = r'(#+\s+Điều\s+\d+[:.].*?)(?=\n+#+\s+Mẫu số|\n+#+\s+Phụ lục|\n+#+\s+Điều\s+\d+|\n+#+\s+Chương\s+[IVXLCDM]+|$)'
            articles = re.findall(article_pattern, pdf_info["content"], re.DOTALL)
            for article in articles:
                page_num = 0
                for muc in muc_luc:
                    if article.find(muc["object"]) != -1:
                        page_num = muc["page"]

                latex_patterns = r'\$\$([\s\S]*?)\$\$'
                latex_matches_origin = re.findall(latex_patterns, article)
                latex_matches = re.findall(latex_patterns, repr(article))
                for i in range(len(latex_matches)):
                    clean_latex = self.convert_latex_to_text(latex=latex_matches[i])
                    article.replace(latex_matches_origin[i], clean_latex)

                article = re.sub(r'\* Trang \d+', '', article)
                article = re.sub(r'\*Begin table\*', ' ', article)
                article = re.sub(r'\*End table\*', ' ', article)
                article = re.sub(r'\n+', '\n', article)
                markdown_special_chars = r'[\\#\*]'
                a_content = re.sub(markdown_special_chars, '', article)

                documents.append({"raw_text": article, "content": a_content, "summary": self.get_chunk_summary(a_content), "src": pdf_info["src"], "chapter": "", "page": page_num, "reference": pdf_info["reference"]})
        else:
            article_pattern = r'(#+\s+Điều\s+\d+[:.].*?)(?=\n+#+\s+Mẫu số|\n+#+\s+Phụ lục|\n+#+\s+Điều\s+\d+|\n+#+\s+Chương\s+[IVXLCDM]+|$)'
            chapter_to_articles = {}
            for chapter in chapters:
                chapter_title_match = re.search(r'##\s+(Chương\s+\w+)', chapter)
                if chapter_title_match:
                    chapter_title = chapter_title_match.group(1)
                    chapter_to_articles[chapter_title] = []
                    
                    articles = re.findall(article_pattern, chapter, re.S)
                    for article in articles:
                        page_num = 0
                        for muc in muc_luc:
                            if article.find(muc["object"]) != -1:
                                page_num = muc["page"]
                        latex_patterns = r'\$\$([\s\S]*?)\$\$'
                        latex_matches_origin = re.findall(latex_patterns, article)
                        latex_matches = re.findall(latex_patterns, repr(article))
                        for i in range(len(latex_matches)):
                            clean_latex = self.convert_latex_to_text(latex=latex_matches[i])
                            article.replace(latex_matches_origin[i], clean_latex)

                        article = re.sub(r'\* Trang \d+', '', article)
                        article = re.sub(r'\*Begin table\*', ' ', article)
                        article = re.sub(r'\*End table\*', ' ', article)
                        article = re.sub(r'\n+', '\n', article)
                        markdown_special_chars = r'[\\#\*]'
                        a_content = re.sub(markdown_special_chars, '', article)
                        for muc in muc_luc:
                            if article.find(muc["object"]) != -1:
                                documents.append({"raw_text": article, "content": a_content, "summary": self.get_chunk_summary(a_content), "src": pdf_info["src"], "chapter": chapter_title, "page": page_num, "reference": pdf_info["reference"]})
                                break
        return documents
    
    def create_form(self, pdf_info):
        """
        
        """
        muc_luc = list()
        documents = []
        
        pattern = r'(\* Trang \d+)(.*?)(?=\* Trang \d+|$)'
        form_pattern = r"#+\s+.*Mẫu số.*\s+\d+[^\n]+"
        matches = re.findall(pattern, pdf_info["content"], re.DOTALL)
        pages = [match[0] + match[1] for match in matches]

        for i, page in enumerate(pages, 1):
            f_matches = re.findall(form_pattern, page)
            for f in f_matches:
                muc_luc.append({"object": f, "page": i})

        form_pattern = r"(#\s*\**Mẫu số \d+/PL[IVXLCDM]+\**\n.*?)(?=#\s*\**Mẫu số|\Z)"
        forms = re.findall(form_pattern, pdf_info["content"], re.DOTALL)
        for form in forms:
            page_num = 0
            for muc in muc_luc:
                if form.find(muc["object"]) != -1:
                    page_num = muc["page"]
            form = re.sub(r'\* Trang \d+', '', form)
            form = re.sub(r'\*Begin table\*', ' ', form)
            form = re.sub(r'\*End table\*', ' ', form)
            form = re.sub(r'\n+', '\n', form)
            markdown_special_chars = r'[\\#\*]'
            f_content = re.sub(markdown_special_chars, '', form)

            documents.append({"raw_text": form, "content": f_content, "summary": self.get_chunk_summary(f_content), "src": pdf_info["src"], "chapter": "", "page": page_num, "reference": pdf_info["reference"]})
        return documents
    
    def create_table(self, tables):
        documents = []
        table_pattern = r"Table \d+\s*(.*?)(?=Table \d+|$)"
        for table_info in tables:
            tbls = re.findall(table_pattern, table_info["content"], re.DOTALL)
            for tbl in tbls:
                documents.append({"raw_text": tbl, "content": tbl, "summary": self.get_chunk_summary(tbl), "src": table_info["src"], "chapter": "", "page": table_info["page"], "reference": table_info["reference"]})

        return documents
    
    def convert_latex_to_text(self, latex):
        latex = re.sub(r'\\text\{(.*?)\}', r'\1', latex)

        latex = latex.replace(r'\left\{', '{')
        latex = latex.replace(r'\right\}', '}')

        latex = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'(\1)/(\2)', latex)

        latex = re.sub(r'\\left\[', '[', latex)
        latex = re.sub(r'\\right\]', ']', latex)
        latex = re.sub(r'\\left\(', '(', latex)
        latex = re.sub(r'\\right\)', ')', latex)

        latex = latex.replace(r'\times', '*')
        latex = latex.replace(r'\div', '/')

        latex = re.sub(r'(\d+)\s*\\,\s*\((.*?)\)', r'\1(\2)', latex)

        latex = latex.replace(r'\%', '%').replace('\\', ' ').strip()

        return latex

    def create_equation(self, pdf_info):
        documents = []
        latex_patterns = r'\$\$([\s\S]*?)\$\$'
        pattern = r'(\* Trang \d+)(.*?)(?=\* Trang \d+|$)'
        matches = re.findall(pattern, pdf_info["content"], re.DOTALL)
        pages = [match[0] + match[1] for match in matches]
        for i, page in enumerate(pages, 1):
            raw_page = repr(page)
            latex_matches = re.findall(latex_patterns, raw_page)
            for latex in latex_matches:
                clean_latex = self.convert_latex_to_text(latex=latex)
                documents.append({"raw_text": latex, "content": clean_latex, "summary": self.get_chunk_summary(clean_latex), "src": pdf_info["src"], "chapter": "", "page": i, "reference": pdf_info["reference"]})
        
        return documents
