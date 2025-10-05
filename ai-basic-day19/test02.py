#加载网页数据
from llama_index.readers.web import SimpleWebPageReader

documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["https://finance.eastmoney.com/a/202502033310108421.html"]
)

print(documents)