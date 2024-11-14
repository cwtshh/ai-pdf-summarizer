from langchain_community.document_loaders.pdf import PyPDFLoader

file_path = "ai-pdf-summarizer/Crepusculo_1o-Cap.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

from langchain_core.prompts import PromptTemplate

prompt_template = """Escreva um longo e detalhado resumo do documento a seguir.
Apenas inclua informações que são partes do documento.
Não inclua suas opiniões ou analises.

Document:
"{document}"
Summary:"""

prompt = PromptTemplate(template=prompt_template)

from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain

llm = ChatOpenAI(
    temperature=0.1,
    model_name="llama3.2:latest",
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

from langchain.chains.combine_documents.stuff import StuffDocumentsChain

stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain, document_variable_name="document"
)

result = stuff_chain.run(docs)

print(result)
