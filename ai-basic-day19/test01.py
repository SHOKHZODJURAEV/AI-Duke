from llama_index.core import SimpleDirectoryReader

#加载本地文档进行解析
# documents = SimpleDirectoryReader(input_dir = "/root/public/projects/demo_21/data",required_exts=[".txt"]).load_data()
#加载某个文档
documents = SimpleDirectoryReader(input_files=["/root/public/projects/demo_21/data/pdf内容研报.pdf"]).load_data()
print(documents)