import asyncio
import hashlib
import hmac
import json
import logging
import ipdb
import random
import re
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from http.client import HTTPSConnection
from typing import List, Optional, Tuple, Type, Union
from dotenv import load_dotenv
import zipfile
load_dotenv()
key=os.environ.get("MINERU_API_KEY")
gptkey=os.environ.get("OPENAI_API_KEY")
import aiohttp
import aiohttp.client_exceptions
import requests
from asyncache import cached as acached
from bs4 import BeautifulSoup
from cachetools import TTLCache, cached
from duckduckgo_search import DDGS, AsyncDDGS

from lagent.actions.base_action import AsyncActionMixin, BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.utils import async_as_completed
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

from lagent.llms import openai
from get_refermodel_prompt import refermodel_prompt_cn

PAPERTITLE=[
    "Longformer: The Long-Document Transformer",
    "Score-Based Generative Modeling through Stochastic Differential Equations",
    "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
    "DeepLabv3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation",
    "BigGAN: Large Scale GAN Training for High Fidelity Natural Image Synthesis",
    "StyleGAN: A Style-Based Generator Architecture for Generative Adversarial Networks",
    "StyleGAN2: Analyzing and Improving the Image Quality of StyleGAN",
    "Neural Ordinary Differential Equations",
    "Momentum Contrast for Unsupervised Visual Representation Learning",
    "SimCLR: A Simple Framework for Contrastive Learning of Visual Representations",
    "BYOL: Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning",
    "SWaV: Swapping Assignments between Views for Self-Supervised Learning",
    "Self-training with Noisy Student improves ImageNet classification",
    "DINO: Emerging Properties in Self-Supervised Vision Transformers",
    "Reformer: The Efficient Transformer",
    "Sparse Transformer",
    "Electra: Pre-training Text Encoders as Discriminators Rather Than Generators",
    "ERNIE 2.0: A Continual Pre-training Framework for Language Understanding",
    "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
    "Universal Transformers",
    "Graph Attention Networks",
    "Capsule Networks",
    "Bidirectional LSTM-CRF Models for Sequence Tagging",
    "Deep Reinforcement Learning with Double Q-learning",
    "Proximal Policy Optimization Algorithms",
    "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor",
    "AlphaGo: Mastering the Game of Go with Deep Neural Networks and Tree Search",
    "AlphaGo Zero: Mastering the game of Go without human knowledge",
    "AlphaFold: Using AI for scientific discovery",
    "U-Net: Convolutional Networks for Biomedical Image Segmentation",
    "UNet++: A Nested U-Net Architecture for Medical Image Segmentation",
    "Improved Regularization of Convolutional Neural Networks with Cutout",
    "Mixup: Beyond Empirical Risk Minimization",
    "Label Smoothing Regularization",
    "Bag of Tricks for Image Classification with Convolutional Neural Networks",
    "Dual Path Networks",
    "ResNeSt: Split-Attention Networks",
    "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
    "One Model to Learn Them All",
    "ViViT: A Video Vision Transformer",
    "SlowFast Networks for Video Recognition",
    "I3D: Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset",
    "DeepSpeech: Scaling up End-to-End Speech Recognition",
    "WaveNet: A Generative Model for Raw Audio",
    "Tacotron: Towards End-to-End Speech Synthesis",
    "Transformer TTS: A Transformer-based Text-to-Speech Model",
    "FastSpeech: Fast, Robust and Controllable Text to Speech",
    "DARTS: Differentiable Architecture Search",
    "ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware",
    "Once for All: Train One Network and Specialize it for Efficient Deployment",
    "On the Convergence of Adam and Beyond",
    "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs",
    "Self-Attention Generative Adversarial Networks",
    "GraphSAGE: Inductive Representation Learning on Large Graphs",
    "Decoupled Weight Decay Regularization",
    "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
    "Learning Image Representations by Completing Damaged Jigsaw Puzzles",
    "A Neural Algorithm of Artistic Style",
    "Perceptual Losses for Real-Time Style Transfer and Super-Resolution",
    "Context Encoders: Feature Learning by Inpainting",
    "Improved Techniques for Training GANs",
    "Self-Supervised Learning of Pretext-Invariant Representations",
    "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour",
    "Universal Language Model Fine-tuning for Text Classification",
    "Attention Augmented Convolutional Networks",
    "Representation Learning with Contrastive Predictive Coding",
    "Learning Deep Features for Discriminative Localization",
    "Matrix Capsules with EM Routing"
]



# 简单的 BaseSearch 类，仅用于继承
class BaseSearch:
    def __init__(self, topk: int, black_list: List[str]):
        self.topk = topk
        self.black_list = black_list

    def _filter_results(self, results: list) -> list:
        filtered = []
        for result in results:
            # 假设结果中第一个元素为链接，如果链接中包含黑名单中的内容则过滤掉
            if not any(black_item in result for black_item in self.black_list):
                filtered.append(result)
        return filtered

class ArxivSearch(BaseSearch):
    def __init__(self,
                 topk: int = 3,
                 black_list: List[str] = [
                     'enoN',
                     'youtube.com',
                     'bilibili.com',
                     'researchgate.net',
                 ],
                 **kwargs):
        self.kwargs = kwargs
        super().__init__(topk, black_list)

    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def search(self, query: str, max_retry: int = 3) -> List[str]:
        for attempt in range(max_retry):
            try:
                xml_response = self._call_arxiv_api(query)
                pdf_links = self._extract_pdf_links(xml_response, query)
                return pdf_links
            except Exception as e:
                logging.exception(str(e))
                warnings.warn(f"Retry {attempt+1}/{max_retry} due to error: {e}")
                time.sleep(random.randint(2, 5))
        raise Exception("Failed to get search results from arXiv after retries.")

    @cached(cache=TTLCache(maxsize=100, ttl=600))
    async def asearch(self, query: str, max_retry: int = 3) -> List[str]:
        for attempt in range(max_retry):
            try:
                xml_response = await self._async_call_arxiv_api(query)
                pdf_links = self._extract_pdf_links(xml_response, query)
                return pdf_links
            except Exception as e:
                logging.exception(str(e))
                warnings.warn(f"Retry {attempt+1}/{max_retry} due to error: {e}")
                await asyncio.sleep(random.randint(2, 5))
        raise Exception("Failed to get search results from arXiv after retries.")

    def _call_arxiv_api(self, query: str) -> str:
        endpoint = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f'ti:"{query}"',
            "start": 0,
            "max_results": self.topk,
            **{key: value for key, value in self.kwargs.items() if value is not None},
        }
        logging.debug(f"Calling arXiv API: {endpoint} with params: {params}")
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.text

    async def _async_call_arxiv_api(self, query: str) -> str:
        endpoint = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f'ti:"{query}"',
            "start": 0,
            "max_results": self.topk,
            **{key: value for key, value in self.kwargs.items() if value is not None},
        }
        logging.debug(f"Async calling arXiv API: {endpoint} with params: {params}")
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint, params=params) as resp:
                resp.raise_for_status()
                return await resp.text()

    def _extract_pdf_links(self, xml_response: str, query: str) -> List[str]:
        try:
            root = ET.fromstring(xml_response)
        except ET.ParseError as e:
            logging.exception("XML parsing error: " + str(e))
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        pdf_links = []
        for entry in entries:
            title_elem = entry.find("atom:title", ns)
            if title_elem is not None:
                title_text = title_elem.text.strip()
                if query.lower() in title_text.lower():
                    # 优先查找有 title="pdf" 的 link 元素
                    pdf_link_elem = entry.find("atom:link[@title='pdf']", ns)
                    if pdf_link_elem is not None:
                        pdf_href = pdf_link_elem.get("href")
                        if pdf_href:
                            pdf_links.append(pdf_href)
                    else:
                        # 如果找不到，尝试通过 entry id 构造 PDF 链接
                        id_elem = entry.find("atom:id", ns)
                        if id_elem is not None:
                            abstract_url = id_elem.text.strip()
                            # 替换 "/abs/" 为 "/pdf/" 并添加 ".pdf"
                            pdf_url = abstract_url.replace("/abs/", "/pdf/") + ".pdf"
                            pdf_links.append(pdf_url)
        # 过滤掉黑名单中的链接
        return self._filter_results(pdf_links)
    
    def pdf2doc(self,pdf_link):

        url='https://mineru.net/api/v4/extract/task'
        header = {
            'Content-Type':'application/json',
            "Authorization":key,
        }
        data = {
            'url':pdf_link,
            'is_ocr':True,
            'language':'en',
            'data_id':pdf_link,
        }

        res = requests.post(url,headers=header,json=data)
        print(res.status_code)
        print(res.json())
        print(res.json()["data"])
        return res.json()["data"]["task_id"]
    
    def checkfinish(self,task_id):
        url = f'https://mineru.net/api/v4/extract/task/{task_id}'
        header = {
            'Content-Type':'application/json',
            "Authorization":key,
        }

        res = requests.get(url, headers=header)
        print(res.status_code)
        print(res.json())
        print(res.json()["data"])
        return res.json()["data"]["state"]
    
    def getresult(self,task_id):
        url = f'https://mineru.net/api/v4/extract/task/{task_id}'
        header = {
            'Content-Type':'application/json',
            "Authorization":key,
        }

        res = requests.get(url, headers=header)
        print(res.status_code)
        print(res.json())
        print(res.json()["data"])
        return res.json()["data"]
    def getpaper(self,url,name):
        file_path=f"/home/qyujiecomputer/qzy/paper/{name}_zip"
        response=requests.get(url,stream=True)
        with open(file_path,"wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"文件已下载并保存为{file_path}")
        unzip_path=f"/home/qyujiecomputer/qzy/paper/{name}"
        with zipfile.ZipFile(file_path,'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        print("文件已解压")
        os.remove(file_path)
        return unzip_path
        

    def getabstract(self, md_file, output_file):
        with open(md_file, "r", encoding="utf-8") as f:
            md_text = f.read()
        pattern = r"(?mi)^#\s*abstract\s*\n+(.*?)(?=^#\s|\Z)"
        
        match = re.search(pattern, md_text, re.DOTALL)
        if match:
            abstract_text = match.group(1).strip()
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("# Abstract\n\n" + abstract_text)
            print(f"Abstract 提取完成，已保存到 {output_file}")
        else:
            print("未找到 Abstract 部分")
    
    def getexperiment(self, md_file, output_file):
        with open(md_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        experiment_lines = []
        in_experiment = False
        experiment_num = None

        header_pattern = re.compile(r"^#\s*(\d+)(?:\.\d+)?\s*(.*)", re.IGNORECASE)
        
        for line in lines:
            header_match = header_pattern.match(line)
            if header_match:
                header_number = header_match.group(1)
                header_text = header_match.group(2)
                if not in_experiment:
                    if re.search(r"experiment", header_text, re.IGNORECASE):
                        in_experiment = True
                        experiment_num = header_number
                        experiment_lines.append("# Experiments\n")
                        continue  # 不重复添加当前标题原文
                else:
                    if header_number != experiment_num:
                        break
            if in_experiment:
                experiment_lines.append(line)
        
        experiment_text = "".join(experiment_lines).strip()
        if experiment_text:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(experiment_text)
            print(f"Experiment 部分提取完成，已保存到 {output_file}")
        else:
            print("未找到 Experiment 部分")


    def getreferences(self, md_file, output_file="references.md"):
        with open(md_file, "r", encoding="utf-8") as f:
            md_text = f.read()

        match = re.search(r"(?mi)^#\s*(?:\d+\.\s*)?(?:reference(?:s)?)\s*\n+(.*?)(?=^#|\Z)", md_text, re.DOTALL)
        if match:
            references_text = match.group(1).strip()

            with open(output_file, "w", encoding="utf-8") as f:
                f.write("# References\n\n" + references_text)

            print(f"References 部分提取完成，已保存到 {output_file}")
        else:
            print("未找到 References 部分")

    def process_query(query):
        print(f"\nProcessing query: {query}")
        link = ArxivSearch.search(query["论文名称"])
        if not link:
            print("No PDF links found.")
            return query  
        link = link + ".pdf"
        print(f"PDF link: {link}")
        task_id = ArxivSearch.pdf2doc(link)
        # 轮询检查任务状态
        while ArxivSearch.checkfinish(task_id) != 'done':
            print("Conversion in progress, waiting 10 seconds...")
            time.sleep(10)
        doc_dic = ArxivSearch.getresult(task_id)
        zip_url = doc_dic.get("full_zip_url", "")
        print(f"DOC zip URL: {zip_url}")
        # 调用 getpaper 和 getabstract 等方法处理结果
        unzip_path = ArxivSearch.getpaper(zip_url, query)
        md_path = unzip_path + "/full.md"
        abstract_path = unzip_path + "/abstract.md"
        ArxivSearch.getabstract(md_path, abstract_path)
        with open(abstract_path, "r", encoding="utf-8") as file:
            abs_text = file.read()
        query["摘要"] = abs_text
        return query

    def getcounterpart(self, cfg):
        # 使用线程池并发处理 cfg 中的所有对象
        with ThreadPoolExecutor() as executor:
            # 提交所有任务
            futures = [executor.submit(ArxivSearch.process_query, query) for query in cfg]
            # 等待所有任务结束并获取结果
            results = [future.result() for future in futures]
        return results

    



def sync_main():
    queries = PAPERTITLE
    arxiv_search = ArxivSearch(topk=5)
    for query in queries:
        try:
            print(f"\nProcessing query: {query}")
            pdf_links = arxiv_search.search(query)
            if not pdf_links:
                print("No PDF links found.")
                continue
            link = pdf_links[0]
            link = link + ".pdf"
            print(f"PDF link: {link}")
            task_id = arxiv_search.pdf2doc(link)
            while arxiv_search.checkfinish(task_id) != 'done':
                print("Conversion in progress, waiting 10 seconds...")
                time.sleep(10)
            doc_dic = arxiv_search.getresult(task_id)
            zip_url = doc_dic.get("full_zip_url", "")
            print(f"DOC zip URL: {zip_url}")
            unzip_path=arxiv_search.getpaper(zip_url,query)
            # unzip_path="/home/qyujiecomputer/qzy/paper/YOLO9000: Better, Faster, Stronger"
            md_path=unzip_path + "/full.md"
            abstract_path=unzip_path + "/abstract.md"
            experiment_path=unzip_path + "/experiment.md"
            reference_path=unzip_path + "/reference.md"
            arxiv_search.getabstract(md_path,abstract_path)
            arxiv_search.getexperiment(md_path,experiment_path)
            arxiv_search.getreferences(md_path,reference_path)
            with open(experiment_path, "r", encoding="utf-8") as file:
                exp = file.read()
                exp = exp.replace("{", "{{").replace("}", "}}")
            with open(reference_path, "r", encoding="utf-8") as file:
                ref = file.read()
                ref = ref.replace("{", "{{").replace("}", "}}")
            counterprompt=refermodel_prompt_cn
            # print(counterprompt)
            api = openai.GPTAPI(model_type='gpt-4o', key=gptkey,json_mode=True)

            messages = [
                {"role": "user", "content": counterprompt}
            ]
            response = api.chat(messages)
            print("回复:", response)
            ctpdic=json.loads(response)
            break

        except Exception as e:
            print(f"Search error (sync) for query '{query}': {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    arxiv_search = ArxivSearch(topk=5)
    query = "A Neural Algorithm of Artistic Style"


    # # 同步调用 arXiv API 并输出返回的 PDF 链接列表
    # try:
    #     pdf_links = arxiv_search.search(query)
    #     print("PDF links (synchronous):")
    #     for link in pdf_links:
    #         print(link)
    # except Exception as e:
    #     print(f"Search error (sync): {e}")

    # 异步调用测试
    async def async_main():
        try:
            pdf_links_async = await arxiv_search.asearch(query)
            print("\nPDF links (asynchronous):")
            link=pdf_links_async[0]
            link = link + ".pdf"
            print(link)
            task_id=(arxiv_search.pdf2doc(link))
            while(arxiv_search.checkfinish(task_id)!='done'):
                time.sleep(10)
                continue
            doc_dic=arxiv_search.getresult(task_id)
            zip_url=doc_dic["full_zip_url"]
            unzip_path=arxiv_search.getpaper(zip_url,query)
            # unzip_path="/home/qyujiecomputer/qzy/paper/Neural Machine Translation by Jointly Learning to Align and Translate"
            md_path=unzip_path + "/full.md"
            abstract_path=unzip_path + "/abstract.md"
            experiment_path=unzip_path + "/experiment.md"
            reference_path=unzip_path + "/reference.md"
            arxiv_search.getabstract(md_path,abstract_path)
            arxiv_search.getexperiment(md_path,experiment_path)
            arxiv_search.getreferences(md_path,reference_path)
            with open(experiment_path, "r", encoding="utf-8") as file:
                exp = file.read()
                exp = exp.replace("{", "{{").replace("}", "}}")
            with open(reference_path, "r", encoding="utf-8") as file:
                ref = file.read()
                ref = ref.replace("{", "{{").replace("}", "}}")
            counterprompt=refermodel_prompt_cn1.format(Experiments=exp,References=ref)
            # print(counterprompt)
            api = openai.GPTAPI(model_type='gpt-4o', key=gptkey,json_mode=True)#R1

            messages = [
                {"role": "user", "content": counterprompt}
            ]
            response = api.chat(messages)
            print("回复:", response)
            ctpdic=json.loads(response)

        except Exception as e:
            print(f"Search error (async): {e}")

    # sync_main()
    asyncio.run(async_main())


# class GitHubSearch(BaseSearch):
#     def __init__(self,
#                  api_key: str,
#                  topk: int = 3,
#                  black_list: List[str] = [
#                      'enoN',
#                      'youtube.com',
#                      'bilibili.com',
#                      'researchgate.net',
#                  ],
#                  **kwargs):
#         self.api_key = api_key
#         self.proxy = kwargs.get('proxy')
#         self.kwargs = kwargs
#         super().__init__(topk, black_list)

#         # Set up logging for debugging
#         logging.basicConfig(level=logging.DEBUG)
#         self.logger = logging.getLogger(__name__)

#     @cached(cache=TTLCache(maxsize=100, ttl=600))
#     def search(self, query: str, max_retry: int = 3) -> dict:
#         for attempt in range(max_retry):
#             try:
#                 self.logger.debug(f"Attempt {attempt + 1}/{max_retry} - Sending request to GitHub API.")
#                 response = self._call_github_api(query)
#                 return self._parse_response(response)
#             except Exception as e:
#                 # Log exception details
#                 self.logger.error(f"Error during API call attempt {attempt + 1}: {str(e)}")
#                 logging.exception("Detailed exception:")
#                 warnings.warn(f'Retry {attempt + 1}/{max_retry} due to error: {e}')
#                 time.sleep(random.randint(2, 5))
#         raise Exception('Failed to get search results from GitHub Search after retries.')

#     @acached(cache=TTLCache(maxsize=100, ttl=600))
#     async def asearch(self, query: str, max_retry: int = 3) -> dict:
#         for attempt in range(max_retry):
#             try:
#                 self.logger.debug(f"Attempt {attempt + 1}/{max_retry} - Sending async request to GitHub API.")
#                 response = await self._async_call_github_api(query)
#                 return self._parse_response(response)
#             except Exception as e:
#                 # Log exception details
#                 self.logger.error(f"Error during API call attempt {attempt + 1}: {str(e)}")
#                 logging.exception("Detailed exception:")
#                 warnings.warn(f'Retry {attempt + 1}/{max_retry} due to error: {e}')
#                 await asyncio.sleep(random.randint(2, 5))
#         raise Exception('Failed to get search results from GitHub Search after retries.')

#     def _call_github_api(self, query: str) -> dict:
#         endpoint = 'https://api.github.com/search/repositories'
#         params = {
#             'q': query,
#             'sort': 'stars',
#             'per_page': 1,
#             'page': 1,
#         }
#         headers = {
#             'User-Agent': 'requests/2.32.3',
#             'Authorization': self.api_key,
#             'Accept': 'application/json'
#         }
#         try:
#             self.logger.debug(f"Request URL: {endpoint}?q={query}&sort=stars")
#             self.logger.debug(f"Request headers: {headers}")
#             response = requests.get(endpoint, headers=headers, params=params, proxies=self.proxy)
#             response.raise_for_status()  # This will raise an exception if the status code is 4xx/5xx
#             self.logger.debug(f"Response status code: {response.status_code}")
#             return response.json()
#         except requests.exceptions.RequestException as e:
#             # Log the error
#             self.logger.error(f"Request failed with error: {e}")
#             raise Exception(f"GitHub API call failed: {str(e)}")

#     async def _async_call_github_api(self, query: str) -> dict:
#         endpoint = 'https://api.github.com/search/repositories'
#         params = {
#             'q': query,
#             'sort': 'stars',
#             'per_page': 1,
#             'page': 1,
#         }
#         headers = {
#             'User-Agent': 'requests/2.32.3',
#             'Authorization': self.api_key,
#             'Accept': 'application/json'
#         }
#         try:
#             self.logger.debug(f"Async Request URL: {endpoint}?q={query}&sort=stars")
#             self.logger.debug(f"Async Request headers: {headers}")
#             async with aiohttp.ClientSession(raise_for_status=True) as session:
#                 async with session.get(endpoint, headers=headers, params=params, proxy=self.proxy) as resp:
#                     self.logger.debug(f"Async Response status code: {resp.status}")
#                     return await resp.json()
#         except aiohttp.ClientError as e:
#             # Log the error
#             self.logger.error(f"Async request failed with error: {e}")
#             raise Exception(f"GitHub API async call failed: {str(e)}")

#     def _parse_response(self, response: dict) -> dict:
#         # Log the raw response for debugging
#         self.logger.debug(f"Response data: {response}")
#         raw_results = []

        # for item in response.get('items', []):
        #     # Log the details of each item
        #     self.logger.debug(f"Processing item: {item}")
        #     url = item.get('html_url')
        #     description = item.get('description', 'No description available')
        #     name = item.get('name')

        #     # Append item details to the results list
        #     raw_results.append((url, description, name))

#         # Filter results and return
#         return self._filter_results(raw_results)
