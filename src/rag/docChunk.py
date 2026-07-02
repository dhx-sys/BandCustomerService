import pandas as pd
from llama_index.core import Document

def load_fault_excel(file_path: str) -> list[Document]:
    df = pd.read_excel(file_path, engine="openpyxl")
    doc_list = []
    for _, row in df.iterrows():
        # 结构化拼接，检索时能匹配任意字段：故障现象、报错码、处理步骤
        content = f"""
【故障编码】{row["故障编码"]}
【诊断结论】{row["诊断结论"]}
【回复话术】{row["回复话术"]}
【分步处理意见】{row["处理意见"]}
        """.strip()
        doc = Document(text=content, metadata={"fault_code": row["故障编码"]})
        doc_list.append(doc)
    return doc_list
#fault_docs = load_fault_excel("./data/光端上云客服诊断话术梳理.xlsx")
