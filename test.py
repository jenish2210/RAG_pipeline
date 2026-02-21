from langchain_openai import OpenAIEmbeddings
emb = OpenAIEmbeddings()
print(emb.embed_query("test"))