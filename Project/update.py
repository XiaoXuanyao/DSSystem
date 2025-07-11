
import os
import sys
import lib



def regenerate_vector_database(embedding_model, files:list[str], chunk_size=512, chunk_overlap=128):
    all_chunks = []
    for file in files:
        filename = file.split("/")[-1].split(".")[0]
        if file.endswith(".pdf"):
            pdf_loader = lib.load_pdf(file)
            chunks = lib.split_pdf(pdf_loader, filename, chunk_size, chunk_overlap)
        elif file.endswith(".docx"):
            docx_loader = lib.load_docx(file)
            chunks = lib.split_docx(docx_loader, filename, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported file type: {file}")
        all_chunks.extend(chunks)
    lib.store_chunks(embedding_model, all_chunks, "runs/dataset1", "collection1")



def update_vector_database(chunk_size=512, chunk_overlap=128):
    """
    更新向量数据库，重新加载所有数据文件并生成向量
    """
    embedding_model = lib.load_embedding_model()
    regenerate_vector_database(embedding_model, [
        "datas/" + file for file in os.listdir("datas") if file[:1] != "~"
    ], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
