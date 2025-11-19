---
layout: post
title: Building an AI Help-Desk Assistant Using Retrieval Augmented Generation (RAG)
image: "/posts/classification-title-img.png"
tags: [GenAI, RAG, LLMs, Python, LangChain]
---

In this project we build a real, production-style AI assistant for **ABC Grocery**, capable of answering customer help-desk questions using **Retrieval Augmented Generation (RAG)**.  

We begin by building a *core RAG system* that loads internal documents, chunks them intelligently, embeds them into a vector database, retrieves relevant content, and generates grounded answers.  

We then extend the assistant by **adding conversational memory**, allowing the model to maintain a short-term personalised dialogue while still respecting strict grounding rules.

# Table of Contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. RAG Overview](#rag-overview)
- [03. Building the Core RAG System](#rag-core)
    - [Secure API Handling](#rag-api)
    - [Document Loading](#rag-docs)
    - [Document Chunking](#rag-chunking)
    - [Embeddings & Vector Store](#rag-embeddings)
    - [LLM Setup](#rag-llm)
    - [Prompt Template](#rag-prompt)
    - [Retriever Setup](#rag-retriever)
    - [Full RAG Pipeline](#rag-pipeline)
- [04. Enhancing the Assistant With Memory](#rag-memory)
- [05. Application & Examples](#rag-application)
- [06. Growth & Next Steps](#growth-next-steps)

___

# 00. Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

ABC Grocery operates a busy customer help-desk, answering queries around store hours, product availability, delivery services, loyalty cards, payments, and general store operations.

The client wants an **AI assistant** that can answer these questions accurately, consistently, and safely, using only approved internal information.

### Actions <a name="overview-actions"></a>

We built a full end-to-end RAG system that:

* loaded internal help-desk documentation  
* split it into meaningful chunks  
* created dense vector embeddings  
* stored these embeddings in a persistent vector database  
* retrieved only the most relevant content at query time  
* generated answers grounded strictly in this retrieved context  

We also extended the project with **conversational memory**, enabling more natural multi-turn interactions while ensuring the assistant never hallucinates.

Internally, we also added monitoring, tracing, and evaluation using LangSmith during development.

### Results <a name="overview-results"></a>

The final assistant:

* reliably answers customer help-desk questions  
* grounds every answer in retrieved internal documentation  
* rejects unsupported questions with a safe fallback message  
* maintains short-term conversational history for better UX  
* prevents hallucinations using strict grounding rules  

### Growth/Next Steps <a name="overview-growth"></a>

Potential future enhancements include:

* ingestion of multiple document types (PDFs, HTML, product catalogues)  
* adding tool use such as SQL lookups for live stock, prices, or loyalty data  
* adding a real chat interface (frontend + backend)  
* streaming responses for improved UX  
* building automated daily document ingestion pipelines  

___

# 01. Data Overview <a name="data-overview"></a>

The dataset contains **many question–answer pairs** taken from ABC Grocery’s internal help-desk documentation.

Each Q&A pair follows a consistent structure, which can be seen below for 5 examples:

```md
### 0001
Q: What is ABC Grocery?
A: ABC Grocery is a family-run supermarket focused on fresh produce, household essentials, and friendly service.

### 0004
Q: What hours are you open on public holidays?
A: Most stores operate reduced hours on public holidays. Please check our store locator for updated hours.

### 0012
Q: Do you offer home delivery?
A: Yes. We offer home delivery 7 days a week. Delivery fees and times depend on location.

### 0020
Q: How do I update my loyalty card details?
A: You can update loyalty details online or by calling our customer support team.

### 0027
Q: Do you sell gluten-free products?
A: Yes. We carry a wide range of gluten-free products across bakery, frozen, snacks, and household aisles.
```

___

# 02. RAG Overview <a name="rag-overview"></a>

Large Language Models are powerful, but they have a key limitation:  
**their knowledge is fixed at training time**, and they cannot reliably retrieve up-to-date, organisation-specific, or policy-specific information.

A naive solution would be to simply **feed the entire help-desk document into the model on every query**, but this has major drawbacks:

* It is slow  
* It is expensive (token costs scale with document length)  
* It overwhelms the model with irrelevant information  
* It dramatically increases the risk of hallucination  
* It doesn’t scale as documents grow into hundreds of pages  

**RAG solves all of these issues.**

With RAG:

1. We embed the documents into a vector database.  
2. When a user asks a question, we retrieve *only the most relevant chunks*.  
3. We pass this small, focused context into the LLM.  
4. The LLM generates a grounded answer based solely on verified internal content.

This ensures answers are **factual, fast, cheap, and controllable**.

___

# 03. Building the Core RAG System <a name="rag-core"></a>

Each subsection below explains both the code *and the concept behind it*.

---

## Secure API Handling <a name="rag-api"></a>

We load API keys from a `.env` file.  
This prevents credentials from being hard-coded directly in the script.

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Document Loading <a name="rag-docs"></a>

We use LangChain’s `TextLoader` to import our help-desk markdown file.

```python
from langchain_community.document_loaders import TextLoader

raw_filename = 'abc-grocery-help-desk-data.md'
loader = TextLoader(raw_filename, encoding="utf-8")
docs = loader.load()
text = docs[0].page_content
```

**Why this matters:**  
Document loaders standardise the data into LangChain `Document` objects, which makes later steps like chunking and embedding seamless.

---

## Document Chunking <a name="rag-chunking"></a>

We split the markdown by level-3 headers (`###`), where each header introduces a new Q&A pair.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("###", "id")],
    strip_headers=True
)

chunked_docs = splitter.split_text(text)
print(len(chunked_docs), "Q/A chunks")
```

**Why this matters:**  
Chunking ensures retrieval focuses on the specific Q&A pair that relates to a user query.  
Good chunking dramatically improves retrieval accuracy.

---

## Embeddings & Vector Store <a name="rag-embeddings"></a>

Embeddings convert text into **numeric vectors** that represent meaning.  
Documents with similar meaning end up closer together in vector space.

We embed each Q&A chunk and store the embeddings in Chroma:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="abc_vector_db_chroma",
    collection_name="abc_help_qa"
)
```

To load later:

```python
vectorstore = Chroma(
    persist_directory="abc_vector_db_chroma",
    collection_name="abc_help_qa",
    embedding_function=embeddings
)
```

---

## LLM Setup <a name="rag-llm"></a>

We instantiate the model that generates the final answer:

```python
from langchain_openai import ChatOpenAI

abc_assistant_llm = ChatOpenAI(model="gpt-5",
                               temperature=0,
                               max_tokens=None,
                               timeout=None,
                               max_retries=1)
```

**Explanation:**

* **model="gpt-5"** — the LLM used for answer generation  
* **temperature=0** — ensures deterministic, factual answers  
* **max_tokens=None** — no manual cap on response length  
* **timeout** and **max_retries** — make execution more robust  

A temperature of 0 is essential for help-desk systems where consistency and accuracy matter more than creativity.

---

## Prompt Template <a name="rag-prompt"></a>

The prompt instructs the model to answer **only** using retrieved context, and to avoid hallucination.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(
"""
System Instructions: You are a helpful assistant for ABC Grocery - your job is to find the best solutions & answers for the customer's query.
Answer ONLY using the provided context. If the answer is not in the context, say that you don't have this information and encourage the customer to email human@abc-grocery.com

Context: {context}

Question: {input}

Answer:
"""
)
```

**Why this matters:**  
Prompt templates are the “instructions” that govern how the LLM behaves.  
They ensure the assistant is safe, grounded, and consistent.

---

## Retriever Setup <a name="rag-retriever"></a>

We configure how relevant chunks are selected from the vector database:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 6, "score_threshold": 0.25}
)
```

**Meaning:**  
* retrieve the top-6 most relevant chunks  
* only return chunks above a relevance threshold  

This keeps the context focused and prevents irrelevant content from confusing the LLM.

---

## Full RAG Pipeline <a name="rag-pipeline"></a>

This pipeline connects all components:

1. take the user query  
2. retrieve relevant chunks  
3. format them  
4. inject them into the prompt  
5. call the LLM  
6. return the answer  

```python
from langchain_core.runnables import RunnableLambda
from operator import itemgetter

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_answer_chain = (
    {
        "context": itemgetter("input") | retriever | RunnableLambda(format_docs),
        "input": itemgetter("input"),
    }
    | prompt_template
    | abc_assistant_llm
)
```

This is the “brain” of the system — the end-to-end mechanism that retrieves and answers.

___

# 04. Enhancing the Assistant With Memory <a name="rag-memory"></a>

In the upgraded version, we introduced **conversational memory**, allowing multi-turn dialogue while still obeying strict grounding rules.

Memory is added through:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
```

And by including a `history` block inside the prompt template.

This keeps conversations natural (e.g., “What about weekends?”) while ensuring that no historical message overrides the authoritative document context.

___

# 05. Application & Examples <a name="rag-application"></a>

Here are some example queries we passed into the system:

```python
query = "What hours are you open on Easter Sunday?"
response = rag_answer_chain.invoke({"input": query})
print(response)
```

```python
query = "Do you offer gluten-free products?"
```

```python
query = "How do I update my loyalty card details?"
```

```python
query = "Can you deliver to rural areas?"
```

```python
query = "Are your deli items suitable for vegetarians?"
```

Below each query, I will manually insert the model’s actual response:

> **[PLACEHOLDER: Insert model output here once captured]**

You can repeat this for any number of examples.

___

# 06. Growth & Next Steps <a name="growth-next-steps"></a>

Potential future enhancements include:

* ingestion of multiple data types (PDFs, HTML, product catalogues, CMS pages)  
* integrating SQL tools for real-time store data, delivery slots, or loyalty information  
* building a production web interface (React + FastAPI)  
* automated indexing pipelines to detect new documents  
* response streaming for real-time chat UX  

This project forms a strong foundation for a scalable enterprise help-desk assistant powered by Retrieval Augmented Generation.

___
