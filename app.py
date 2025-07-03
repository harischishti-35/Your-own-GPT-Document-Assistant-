
# import os
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings



# from dotenv import load_dotenv
# load_dotenv()

# ## load the Groq API key
# groq_api_key=os.environ['GROQ_API_KEY']
# os.environ["GOOGLE_API_KEY"]=os.getenv('GOOGLE_API_KEY')

# st.title("Harry's Chat")

# llm=ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")


# prompt=ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}

# """
# )


# def vector_embedding():
    
#     if "vectors" not in st.session_state:

#         st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-001')
#         st.session_state.loader=PyPDFDirectoryLoader("./lawpdfs") ## Data Ingestion
#         st.session_state.docs=st.session_state.loader.load() ## Document Loading
#         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
#         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs) #splitting
#         st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings


# prompt1=st.text_input("What you want to ask from the documents?" )

# if st.button("Creating Vector Store"):
#     vector_embedding()
#     st.write("Vector Store DB IS Ready")

# import time


# if prompt1:
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     retriever = st.session_state.vectors.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    
#     start=time.process_time()
#     response=retrieval_chain.invoke({"input":prompt1})
#     print("Response time :",time.process_time()-start)
#     st.write(response['answer'])

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")
    
    







# Urdu to English translation




# import os
# import time
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_groq import ChatGroq
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# # Load environment variables
# load_dotenv()

# # Load API keys
# groq_api_key = os.environ['GROQ_API_KEY']
# os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

# # Streamlit app title
# st.title("Harry's Chat")

# # Initialize the Groq language model
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# # Define the prompt template for answering questions
# prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response based on the question
#     <context>
#     {context}
#     <context>
#     Questions:{input}
#     """
# )

# def vector_embedding():
#     """Function to load documents, split them into chunks, and create vector embeddings."""
#     if "vectors" not in st.session_state:
#         with st.spinner("Creating vector embeddings..."):
#             st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
#             st.session_state.loader = PyPDFDirectoryLoader("./lawpdfs")  # Data Ingestion
#             st.session_state.docs = st.session_state.loader.load()  # Document Loading
#             st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
#             st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
#             st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings
#         st.success("Vector embeddings created successfully!")

# # User input for questions
# prompt1 = st.text_input("What you want to ask from the documents?")

# # Button to create vector store
# if st.button("Create Vector Store"):
#     vector_embedding()
#     st.write("Vector Store DB is ready.")

# # Process the question and display the answer
# if prompt1:
#     if "vectors" not in st.session_state:
#         st.error("Please create the vector store first.")
#     else:
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
#         start = time.process_time()
#         response = retrieval_chain.invoke({"input": prompt1})
#         st.write(f"Response time: {time.process_time() - start:.2f} seconds")
#         st.write("**Answer:**")
#         st.write(response['answer'])

#         Add a button to translate the answer to Urdu
#         if st.button("Translate to Urdu"):
#             with st.spinner("Translating to Urdu..."):
#                 # Use the Groq language model to translate the answer
#                 translation_response = llm.invoke(f"Translate the following English text to Urdu: {response['answer']}")
#                 st.success("Translation Complete!")
#                 st.write("**Translated Answer (Urdu):**")
#                 st.write(translation_response.content)

#         # Display relevant document chunks
#         with st.expander("Document Similarity Search"):
#             for i, doc in enumerate(response["context"]):
#                 st.write(doc.page_content)
#                 st.write("--------------------------------")














# Embedding database save changes


# import os
# import time
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_groq import ChatGroq
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from tenacity import retry, stop_after_attempt, wait_exponential

# # Load environment variables
# load_dotenv()

# # Load API keys
# groq_api_key = os.environ['GROQ_API_KEY']
# os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

# # Streamlit app title
# st.title("Harry's Chat")

# # Initialize the Groq language model
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# # Define the prompt template
# prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response based on the question
#     <context>
#     {context}
#     <context>
#     Questions:{input}
#     """
# )

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def vector_embedding():
#     """Function to load documents, split them into chunks, and create vector embeddings."""
#     if "vectors" not in st.session_state:
#         embeddings_file = "faiss_embeddings.index"  # File to save/load embeddings

#         if os.path.exists(embeddings_file):
#             # Load embeddings from disk
#             with st.spinner("Loading vector embeddings from disk..."):
#                 st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', timeout=120)
#                 st.session_state.vectors = FAISS.load_local(
#                     embeddings_file,
#                     st.session_state.embeddings,
#                     allow_dangerous_deserialization=True  # Enable deserialization
#                 )
#             st.success("Vector embeddings loaded successfully!")
#         else:
#             # Create new embeddings and save to disk
#             with st.spinner("Creating vector embeddings..."):
#                 st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', timeout=120)
#                 st.session_state.loader = PyPDFDirectoryLoader("./lawpdfs")  # Data Ingestion
#                 st.session_state.docs = st.session_state.loader.load()  # Document Loading
#                 st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
#                 st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
#                 st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings
#                 st.session_state.vectors.save_local(embeddings_file)  # Save embeddings to disk
#             st.success("Vector embeddings created and saved successfully!")

# # User input for questions
# prompt1 = st.text_input("What you want to ask from the documents?")

# # Button to create vector store
# if st.button("Create Vector Store"):
#     vector_embedding()
#     st.write("Vector Store DB is ready.")

# # Process the question and display the answer
# if prompt1:
#     if "vectors" not in st.session_state:
#         st.error("Please create the vector store first.")
#     else:
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
#         start = time.process_time()
#         response = retrieval_chain.invoke({"input": prompt1})
#         st.write(f"Response time: {time.process_time() - start:.2f} seconds")
#         st.write(response['answer'])
        
#         # Add a button to translate the answer to Urdu
#         if st.button("Translate to Urdu"):
#             with st.spinner("Translating to Urdu..."):
#                 # Use the Groq language model to translate the answer
#                 translation_response = llm.invoke(f"Translate the following English text to Urdu: {response['answer']}")
#                 st.success("Translation Complete!")
#                 st.write("**Translated Answer (Urdu):**")
#                 st.write(translation_response.content)

#         # Display relevant document chunks
#         with st.expander("Document Similarity Search"):
#             for i, doc in enumerate(response["context"]):
#                 st.write(doc.page_content)
#                 st.write("--------------------------------")


# UI changes for 

# import os
# import time
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_groq import ChatGroq
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from tenacity import retry, stop_after_attempt, wait_exponential

# # Load environment variables
# load_dotenv()

# # Load API keys
# groq_api_key = os.environ['GROQ_API_KEY']
# os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

# # Streamlit app title and description
# st.set_page_config(page_title="Harry's Chat", page_icon="🤖", layout="centered")
# st.title("🤖 AI Personal Lawyer")
# st.markdown("""
#     Welcome to **AI Personal Lawyer**! This app allows you to ask questions about your documents and get accurate answers powered by AI.
#     """)

# # Initialize the Groq language model
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# # Define the prompt template
# prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response based on the question
#     <context>
#     {context}
#     <context>
#     Questions:{input}
#     """
# )

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def vector_embedding():
#     """Function to load documents, split them into chunks, and create vector embeddings."""
#     if "vectors" not in st.session_state:
#         embeddings_file = "faiss_embeddings.index"  # File to save/load embeddings

#         if os.path.exists(embeddings_file):
#             # Load embeddings from disk
#             with st.spinner("Loading vector embeddings from disk..."):
#                 st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', timeout=120)
#                 st.session_state.vectors = FAISS.load_local(
#                     embeddings_file,
#                     st.session_state.embeddings,
#                     allow_dangerous_deserialization=True  # Enable deserialization
#                 )
#             st.success("Vector embeddings loaded successfully!")
#         else:
#             # Create new embeddings and save to disk
#             with st.spinner("Creating vector embeddings..."):
#                 st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', timeout=120)
#                 st.session_state.loader = PyPDFDirectoryLoader("./lawpdfs")  # Data Ingestion
#                 st.session_state.docs = st.session_state.loader.load()  # Document Loading
#                 st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
#                 st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
#                 st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings
#                 st.session_state.vectors.save_local(embeddings_file)  # Save embeddings to disk
#             st.success("Vector embeddings created and saved successfully!")

# # Sidebar for navigation and options
# with st.sidebar:
#     st.header("Options")
#     if st.button("Create Vector Store"):
#         vector_embedding()
#     st.markdown("---")
#     st.markdown("### Instructions")
#     st.markdown("""
#         1. Click **Create Vector Store** to load and process your documents.
#         2. Enter your question in the input box below.
#         3. View the answer and relevant document chunks.
#         """)
#     st.markdown("---")
#     st.markdown("### Theme")
#     dark_mode = st.checkbox("Enable Dark Mode")

# # Apply dark mode if enabled
# if dark_mode:
#     st.markdown(
#         """
#         <style>
#         .stApp {
#             background-color: #1e1e1e;
#             color: #ffffff;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # User input for questions
# st.markdown("---")
# st.header("Ask a Question")
# prompt1 = st.text_input("Enter your question here:")

# # Process the question and display the answer
# if prompt1:
#     if "vectors" not in st.session_state:
#         st.error("Please create the vector store first.")
#     else:
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
#         with st.spinner("Generating answer..."):
#             start = time.process_time()
#             response = retrieval_chain.invoke({"input": prompt1})
#             st.write(f"**Response time:** {time.process_time() - start:.2f} seconds")
#             st.markdown("---")
#             st.subheader("Answer")
#             st.write(response['answer'])

            
#             # Add a button to translate the answer to Urdu
#             if st.button("Translate to Urdu"):
#                 with st.spinner("Translating to Urdu..."):
#                     # Use the Groq language model to translate the answer
#                     translation_response = llm.invoke(f"Translate the following English text to Urdu: {response['answer']}")
#                     st.success("Translation Complete!")
#                     st.write("**Translated Answer (Urdu):**")
#                     st.write(translation_response.content)

#             # Display relevant document chunks
#             with st.expander("View Relevant Document Chunks"):
#                 for i, doc in enumerate(response["context"]):
#                     st.markdown(f"**Chunk {i+1}**")
#                     st.write(doc.page_content)
#                     st.markdown("---")


# more ui changes
# Working fine 27-June-25  12:10 pm PIA IT building

# import os
# import time
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_groq import ChatGroq
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from tenacity import retry, stop_after_attempt, wait_exponential

# # Load environment variables
# load_dotenv()

# # Load API keys
# groq_api_key = os.environ['GROQ_API_KEY']
# os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

# # Streamlit page configuration
# st.set_page_config(page_title="PIA ChatBot", page_icon="⚖️", layout="centered")

# st.markdown("""
#     <h1 style='text-align: center;'>PIA Document Assistant</h1>
#     <p style='text-align: center;'>Ask questions about your documents and get answers instantly.</p>
# """, unsafe_allow_html=True)

# # Initialize LLM
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# # Prompt Template
# prompt = ChatPromptTemplate.from_template("""
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Question: {input}
# """)

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def vector_embedding():
#     """Load documents, split into chunks, and create embeddings."""
#     if "vectors" not in st.session_state:
#         embeddings_file = "faiss_embeddings.index"

#         if os.path.exists(embeddings_file):
#             with st.spinner("Loading vector embeddings from disk..."):
#                 st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', timeout=120)
#                 st.session_state.vectors = FAISS.load_local(
#                     embeddings_file,
#                     st.session_state.embeddings,
#                     allow_dangerous_deserialization=True
#                 )
#             st.success("Vector embeddings loaded successfully!")
#         else:
#             with st.spinner("Creating vector embeddings..."):
#                 st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', timeout=120)
#                 st.session_state.loader = PyPDFDirectoryLoader("./lawpdfs")
#                 st.session_state.docs = st.session_state.loader.load()
#                 st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#                 st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
#                 st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
#                 st.session_state.vectors.save_local(embeddings_file)
#             st.success("Vector embeddings created and saved successfully!")

# # Sidebar
# with st.sidebar:
#     st.markdown("### 📁 Navigation")
#     if st.button("📂 Create Vector Store"):
#         vector_embedding()

#     st.markdown("---")
#     st.markdown("### ℹ️ Instructions")
#     st.markdown("""
#     1. Upload your documents.  
#     2. Create the vector store.  
#     3. Ask any question!  
#     4. Translate answer if needed.
#     """)

#     st.markdown("---")
#     dark_mode = st.checkbox("🌑 Enable Dark Mode")

# # Dark mode styling
# if dark_mode:
#     st.markdown("""
#         <style>
#             .stApp {
#                 background-color: #121212;
#                 color: #ffffff;
#             }
#             textarea, input, .stTextInput, .stTextArea, .stButton {
#                 background-color: #1f1f1f !important;
#                 color: #ffffff !important;
#                 border: 1px solid #444444;
#             }
#             .stTextInput>div>div>input {
#                 background-color: #1f1f1f !important;
#                 color: white !important;
#             }
#             .stTextArea>div>textarea {
#                 background-color: #1f1f1f !important;
#                 color: white !important;
#             }
#         </style>
#     """, unsafe_allow_html=True)

# # Main UI
# st.markdown("---")
# st.markdown("### 📝 Ask Your Question")
# prompt1 = st.text_area("🔎 Type your legal question below:", height=150)

# col1, col2 = st.columns(2)

# with col1:
#     submit = st.button("✅ Get Answer")
# with col2:
#     translate = st.button("🌐 Translate to Urdu")

# # Answer generation
# if submit and prompt1:
#     if "vectors" not in st.session_state:
#         st.error("Please create the vector store first.")
#     else:
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)

#         with st.spinner("Generating answer..."):
#             start = time.process_time()
#             response = retrieval_chain.invoke({"input": prompt1})
#             st.write(f"**Response time:** {time.process_time() - start:.2f} seconds")
#             st.markdown("---")
#             st.subheader("📌 Answer")
#             st.write(response['answer'])

#             # Store response in session state for translation
#             st.session_state.last_response = response['answer']
#             st.session_state.context_docs = response["context"]

# # Translation
# if translate and st.session_state.get("last_response"):
#     with st.spinner("Translating to Urdu..."):
#         translated = llm.invoke(f"Translate the following English text to Urdu: {st.session_state.last_response}")
#         st.success("Translation Complete!")
#         st.write("**🌍 Translated Answer (Urdu):**")
#         st.write(translated.content)

# # Document chunks
# if st.session_state.get("context_docs"):
#     with st.expander("📄 View Relevant Document Chunks"):
#         for i, doc in enumerate(st.session_state.context_docs):
#             st.markdown(f"**Chunk {i+1}**")
#             st.write(doc.page_content)
#             st.markdown("---")
            
            
            

# Improvement using Trae Ai

# import os
# import time
# import uuid
# import logging
# from datetime import datetime, timedelta
# from functools import wraps
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_groq import ChatGroq
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from tenacity import retry, stop_after_attempt, wait_exponential

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# def load_api_keys():
#     """Load and validate API keys."""
#     if not os.getenv('GROQ_API_KEY') or not os.getenv('GOOGLE_API_KEY'):
#         st.error("Missing required API keys. Please check your .env file.")
#         st.stop()
#     return os.getenv('GROQ_API_KEY'), os.getenv('GOOGLE_API_KEY')

# # Load API keys
# groq_api_key, google_api_key = load_api_keys()
# os.environ["GOOGLE_API_KEY"] = google_api_key

# # Initialize LLM
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# # Prompt Template
# prompt = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate.from_template("You are a helpful assistant that answers questions based on the provided context."),
#     HumanMessagePromptTemplate.from_template("Here is the context:\n\n{context}\n\nQuestion: {question}"),
#     MessagesPlaceholder(variable_name="chat_history"),
#     HumanMessagePromptTemplate.from_template("{question}")
# ])

# # Rate limiting decorator
# def rate_limit(seconds=60, limit=10):
#     def decorator(func):
#         calls = []
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             now = datetime.now()
#             calls[:] = [call for call in calls if call > now - timedelta(seconds=seconds)]
#             if len(calls) >= limit:
#                 st.error(f"Rate limit exceeded. Please try again in {seconds} seconds.")
#                 return None
#             calls.append(now)
#             return func(*args, **kwargs)
#         return wrapper
#     return decorator

# # Performance tracking decorator
# def track_performance(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start = time.perf_counter()
#         result = func(*args, **kwargs)
#         duration = time.perf_counter() - start
#         st.session_state.metrics = st.session_state.get('metrics', []) + [duration]
#         return result
#     return wrapper

# # Input validation
# def validate_input(prompt):
#     if len(prompt.strip()) < 10:
#         st.warning("Please enter a more detailed question.")
#         return False
#     return True

# # Document processing with progress tracking
# def process_documents(docs):
#     progress_bar = st.progress(0)
#     total = len(docs)
#     processed_docs = []
    
#     for i, doc in enumerate(docs):
#         processed_docs.append(doc)
#         progress_bar.progress((i + 1) / total)
#     return processed_docs

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# @st.cache_data(ttl=3600)
# def vector_embedding():
#     """Load documents, split into chunks, and create embeddings with caching."""
#     try:
#         if "vectors" not in st.session_state:
#             embeddings_file = "faiss_embeddings.index"

#             if os.path.exists(embeddings_file):
#                 with st.spinner("Loading vector embeddings from disk..."):
#                     st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
#                         model='models/embedding-001',
#                         timeout=120
#                     )
#                     st.session_state.vectors = FAISS.load_local(
#                         embeddings_file,
#                         st.session_state.embeddings,
#                         allow_dangerous_deserialization=True
#                     )
#                 st.success("Vector embeddings loaded successfully!")
#             else:
#                 with st.spinner("Creating vector embeddings..."):
#                     st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
#                         model='models/embedding-001',
#                         timeout=120
#                     )
#                     st.session_state.loader = PyPDFDirectoryLoader("./lawpdfs")
#                     st.session_state.docs = st.session_state.loader.load()
#                     st.session_state.text_splitter = RecursiveCharacterTextSplitter(
#                         chunk_size=500,
#                         chunk_overlap=50,
#                         length_function=len
#                     )
#                     st.session_state.final_documents = process_documents(
#                         st.session_state.text_splitter.split_documents(st.session_state.docs)
#                     )
#                     st.session_state.vectors = FAISS.from_documents(
#                         st.session_state.final_documents,
#                         st.session_state.embeddings
#                     )
#                     st.session_state.vectors.save_local(embeddings_file)
#                 st.success("Vector embeddings created and saved successfully!")
#     except Exception as e:
#         logger.error(f"Error in vector embedding: {str(e)}")
#         st.error("An error occurred while processing the documents. Please try again.")
#         raise

# @rate_limit(seconds=60, limit=10)
# @track_performance
# def generate_answer(prompt_text):
#     """Generate answer from the document context."""
#     try:
#         if not validate_input(prompt_text):
#             return None

#         if "vectors" not in st.session_state:
#             st.error("Please upload documents first.")
#             return None

#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)

#         response = retrieval_chain.invoke({"question": prompt_text})
#         st.session_state.query_count += 1
#         return response['answer']

#     except Exception as e:
#         logger.error(f"Error generating answer: {str(e)}")
#         raise

# # Streamlit page configuration
# st.set_page_config(
#     page_title="Document ChatBot",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .stApp {
#         background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%);
#     }
#     .main {
#         background-color: white;
#         border-radius: 10px;
#         padding: 20px;
#         margin: 10px;
#     }
#     .stButton>button {
#         background-color: #4CAF50;
#         color: white;
#         border-radius: 20px;
#         padding: 10px 24px;
#         border: none;
#     }
#     .upload-section {
#         border: 2px dashed #ccc;
#         border-radius: 10px;
#         padding: 20px;
#         text-align: center;
#         margin: 20px 0;
#     }
#     .feature-box {
#         background-color: rgba(255, 255, 255, 0.1);
#         border-radius: 10px;
#         padding: 15px;
#         margin: 10px 0;
#         text-align: center;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Header
# st.markdown("""
#     <h1 style='text-align: center; color: white;'>🤖 Document ChatBot</h1>
#     <p style='text-align: center; color: white;'>Ask intelligent questions about your documents and get instant, accurate answers</p>
# """, unsafe_allow_html=True)

# # Main content area
# col1, col2 = st.columns([1, 2])

# with col1:
#     st.markdown("""
#     <div class='upload-section'>
#         <h3>📁 Upload Documents</h3>
#         <p>Support: PDF, TXT, DOC, DOCX • Max 10MB each</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Feature boxes
#     st.markdown("""
#     <div class='feature-box'>
#         <h4>🔍 Smart Search</h4>
#     </div>
#     <div class='feature-box'>
#         <h4>⚡ Instant Answers</h4>
#     </div>
#     <div class='feature-box'>
#         <h4>🔒 Secure & Private</h4>
#     </div>
#     <div class='feature-box'>
#         <h4>📝 Source Citations</h4>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     # Chat interface
#     st.markdown("### 💬 Chat with your documents")
    
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#         st.session_state.query_count = 0

#     # Display chat messages
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Chat input
#     if prompt := st.chat_input("Upload documents to start chatting..."):
#         if "vectors" not in st.session_state:
#             st.error("Please upload documents first.")
#         else:
#             # Add user message to chat history
#             st.session_state.messages.append({"role": "user", "content": prompt})
#             with st.chat_message("user"):
#                 st.markdown(prompt)

#             # Generate response
#             with st.chat_message("assistant"):
#                 with st.spinner("Thinking..."):
#                     try:
#                         response = generate_answer(prompt)
#                         if response:
#                             st.session_state.messages.append({"role": "assistant", "content": response})
#                             st.markdown(response)
#                     except Exception as e:
#                         st.error("Failed to generate response. Please try again.")

# # Sidebar
# with st.sidebar:
#     st.markdown("### 🛠️ Tools")
#     if st.button("🔄 Create Vector Store"):
#         vector_embedding()

#     st.markdown("---")
#     st.markdown("### 📊 Stats")
#     if st.session_state.get('metrics'):
#         st.metric("Total Queries", st.session_state.query_count)
#         avg_time = sum(st.session_state.metrics) / len(st.session_state.metrics)
#         st.metric("Avg Response Time", f"{avg_time:.2f}s")



# Adding new feature download pdfs


# import os
# import time
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_groq import ChatGroq
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from tenacity import retry, stop_after_attempt, wait_exponential

# # Load environment variables
# load_dotenv()

# # Load API keys
# groq_api_key = os.environ['GROQ_API_KEY']
# os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

# # Streamlit page configuration
# st.set_page_config(page_title="PIA ChatBot", page_icon="⚖️", layout="centered")

# st.markdown("""
#     <h1 style='text-align: center;'>PIA Document Assistant</h1>
#     <p style='text-align: center;'>Ask questions about your documents and get answers instantly.</p>
# """, unsafe_allow_html=True)

# # Initialize LLM
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# # Prompt Template
# prompt = ChatPromptTemplate.from_template("""
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question.
# <context>
# {context}
# </context>
# Question: {input}
# """)

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def vector_embedding():
#     """Load PDFs manually, inject metadata, split and embed."""
#     if "vectors" not in st.session_state:
#         embeddings_file = "faiss_embeddings.index"

#         if os.path.exists(embeddings_file):
#             with st.spinner("Loading vector embeddings from disk..."):
#                 st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', timeout=120)
#                 st.session_state.vectors = FAISS.load_local(
#                     embeddings_file,
#                     st.session_state.embeddings,
#                     allow_dangerous_deserialization=True
#                 )
#             st.success("Vector embeddings loaded successfully!")
#         else:
#             with st.spinner("Creating vector embeddings..."):
#                 st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-exp-03-07', timeout=120)

#                 docs = []
#                 pdf_folder = "./lawpdfs"

#                 for filename in os.listdir(pdf_folder):
#                     if filename.endswith(".pdf"):
#                         path = os.path.join(pdf_folder, filename)
#                         loader = PyPDFLoader(path)
#                         pages = loader.load()

#                         for page in pages:
#                             page.metadata["source_file"] = filename
#                             page.metadata["download_url"] = f"/static/lawpdfs/{filename}"

#                         docs.extend(pages)

#                 st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#                 st.session_state.final_documents = st.session_state.text_splitter.split_documents(docs)
#                 st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
#                 st.session_state.vectors.save_local(embeddings_file)
#             st.success("Vector embeddings created and saved successfully!")

# # Sidebar
# with st.sidebar:
#     st.markdown("### 📁 Navigation")
#     if st.button("📂 Create Vector Store"):
#         vector_embedding()

#     st.markdown("---")
#     st.markdown("### ℹ️ Instructions")
#     st.markdown("""
#     1. Place PDFs in `./lawpdfs` and copy them to `./static/lawpdfs`.  
#     2. Click **Create Vector Store**.  
#     3. Ask your legal question.  
#     4. Optionally translate the answer to Urdu.
#     """)

#     st.markdown("---")
#     dark_mode = st.checkbox("🌑 Enable Dark Mode")

# # Dark mode styling
# if dark_mode:
#     st.markdown("""
#         <style>
#             .stApp {
#                 background-color: #121212;
#                 color: #ffffff;
#             }
#             textarea, input, .stTextInput, .stTextArea, .stButton {
#                 background-color: #1f1f1f !important;
#                 color: #ffffff !important;
#                 border: 1px solid #444444;
#             }
#             .stTextInput>div>div>input {
#                 background-color: #1f1f1f !important;
#                 color: white !important;
#             }
#             .stTextArea>div>textarea {
#                 background-color: #1f1f1f !important;
#                 color: white !important;
#             }
#         </style>
#     """, unsafe_allow_html=True)

# # Main UI
# st.markdown("---")
# st.markdown("### 📝 Ask Your Question")
# prompt1 = st.text_area("🔎 Type your legal question below:", height=150)

# col1, col2 = st.columns(2)

# with col1:
#     submit = st.button("✅ Get Answer")
# with col2:
#     translate = st.button("🌐 Translate to Urdu")

# # Answer generation
# if submit and prompt1:
#     if "vectors" not in st.session_state:
#         st.error("Please create the vector store first.")
#     else:
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)

#         with st.spinner("Generating answer..."):
#             start = time.process_time()
#             response = retrieval_chain.invoke({"input": prompt1})
#             st.write(f"**Response time:** {time.process_time() - start:.2f} seconds")
#             st.markdown("---")
#             st.subheader("📌 Answer")
#             st.write(response['answer'])

#             st.session_state.last_response = response['answer']
#             st.session_state.context_docs = response["context"]

#             # Show document references
#             st.subheader("📚 Source References")
#             for doc in st.session_state.context_docs:
#                 filename = doc.metadata.get("source_file", "Unknown Document")
#                 link = doc.metadata.get("download_url", "#")
#                 st.markdown(f"- [{filename}]({link})")

# # Translation
# if translate and st.session_state.get("last_response"):
#     with st.spinner("Translating to Urdu..."):
#         translated = llm.invoke(f"Translate the following English text to Urdu: {st.session_state.last_response}")
#         st.success("Translation Complete!")
#         st.write("**🌍 Translated Answer (Urdu):**")
#         st.write(translated.content)

# # Document chunks
# if st.session_state.get("context_docs"):
#     with st.expander("📄 View Relevant Document Chunks"):
#         for i, doc in enumerate(st.session_state.context_docs):
#             st.markdown(f"**Chunk {i+1}**")
#             st.write(doc.page_content)
#             st.markdown("---")

# 
# ############################################################################################

# import os
# import time
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_groq import ChatGroq
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# # Using the more efficient directory loader from your new script
# from langchain_community.document_loaders import PyPDFDirectoryLoader 
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from tenacity import retry, stop_after_attempt, wait_exponential

# # Load environment variables
# load_dotenv()

# # Load API keys
# groq_api_key = os.environ.get('GROQ_API_KEY')
# google_api_key = os.environ.get('GOOGLE_API_KEY')

# if not groq_api_key or not google_api_key:
#     st.error("API keys for Groq and Google are not set. Please check your .env file.")
#     st.stop()

# os.environ["GOOGLE_API_KEY"] = google_api_key

# # --- Page Configuration (from new UI) ---
# st.set_page_config(page_title="PIA ChatBot", page_icon="⚖️", layout="centered")

# # --- Main Title (from new UI) ---
# st.markdown("""
#     <h1 style='text-align: center;'>PIA Document Assistant</h1>
#     <p style='text-align: center;'>Ask questions about your documents and get answers instantly.</p>
# """, unsafe_allow_html=True)


# # --- Initialize LLM and Session State ---
# if "llm" not in st.session_state:
#     st.session_state.llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# # Initialize all necessary session state keys to avoid errors
# for key in ["last_response", "translated_answer", "response_time", "vectors", "context_docs"]:
#     if key not in st.session_state:
#         st.session_state[key] = None

# # --- FUNCTION DEFINITION SECTION ---
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def vector_embedding():
#     FAISS_INDEX_PATH = "faiss_vector_store"
    
#     if st.session_state.vectors is None:
#         try:
#             embeddings_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
#         except Exception as e:
#             st.error(f"Failed to initialize embeddings model: {e}")
#             return

#         if os.path.exists(FAISS_INDEX_PATH):
#             with st.spinner("Loading existing Vector Store from disk..."):
#                 try:
#                     st.session_state.vectors = FAISS.load_local(
#                         FAISS_INDEX_PATH, 
#                         embeddings_model,
#                         allow_dangerous_deserialization=True
#                     )
#                     st.success("Vector Store loaded successfully!")
#                 except Exception as e:
#                     st.error(f"Error loading existing vector store: {e}")
#             return

#         with st.spinner("Creating new Vector Store. This may take a moment..."):
#             pdf_folder = "./static" # Your PDFs should be in a 'static' folder
#             if not os.path.isdir(pdf_folder) or not os.listdir(pdf_folder):
#                 st.error(f"The '{pdf_folder}' directory is empty or does not exist. Please add PDF files to it.")
#                 return
#             try:
#                 # Using PyPDFDirectoryLoader for efficiency
#                 loader = PyPDFDirectoryLoader(pdf_folder)
#                 docs = loader.load()
#                 if not docs:
#                     st.warning("No PDF documents were found to process.")
#                     return
                
#                 text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#                 final_documents = text_splitter.split_documents(docs)
#                 st.session_state.vectors = FAISS.from_documents(final_documents, embeddings_model)
#                 st.session_state.vectors.save_local(FAISS_INDEX_PATH)
#                 st.success("New Vector Store created and saved successfully!")
#             except Exception as e:
#                 st.error(f"An error occurred during vector embedding creation: {e}")
#     else:
#         st.success("Vector Store is already loaded.")

# # --- Sidebar (from new UI) ---
# with st.sidebar:
#     st.markdown("### 📁 Navigation")
#     if st.button("📂 Create Vector Store"):
#         vector_embedding()

#     st.markdown("---")
#     st.markdown("### ℹ️ Instructions")
#     st.info("""
#     1. Place PDFs in the `./static` folder.
#     2. Click the button above to process.
#     3. Ask any question!
#     4. Translate the answer if needed.
#     """)

#     st.markdown("---")
#     dark_mode = st.checkbox("🌑 Enable Dark Mode")

# # --- Dark Mode Styling (from new UI) ---
# if dark_mode:
#     st.markdown("""
#         <style>
#             .stApp {
#                 background-color: #121212;
#                 color: #ffffff;
#             }
#             /* Styling for input widgets */
#             textarea, input, .stTextInput, .stTextArea {
#                 background-color: #1f1f1f !important;
#                 color: #ffffff !important;
#                 border: 1px solid #444444;
#             }
#             .stTextInput>div>div>input, .stTextArea>div>textarea {
#                 background-color: #1f1f1f !important;
#                 color: white !important;
#             }
#             /* Styling for buttons */
#             .stButton>button {
#                 background-color: #333333;
#                 color: #ffffff;
#                 border: 1px solid #555555;
#             }
#             /* Styling for expander */
#             .stExpander {
#                 background-color: #1f1f1f;
#                 border: 1px solid #444444;
#             }
#         </style>
#     """, unsafe_allow_html=True)

# # --- Main Content Area (from new UI) ---
# st.markdown("---")
# st.markdown("### 📝 Ask Your Question")
# prompt1 = st.text_area("🔎 Type your legal question below:", height=150, label_visibility="collapsed")

# col1, col2 = st.columns(2)
# with col1:
#     submit = st.button("✅ Get Answer")
# with col2:
#     translate = st.button("🌐 Translate to Urdu")

# # --- LOGIC FOR ANSWERING AND TRANSLATING ---
# if submit and prompt1:
#     if st.session_state.vectors is None:
#         st.error("Please create the vector store first using the button in the sidebar.")
#     else:
#         prompt_template = ChatPromptTemplate.from_template(
#             "Answer the questions based on the provided context only.\n"
#             "Please provide the most accurate response based on the question.\n\n"
#             "<context>{context}</context>\n\n"
#             "Question: {input}"
#         )
#         with st.spinner("Generating answer..."):
#             try:
#                 document_chain = create_stuff_documents_chain(st.session_state.llm, prompt_template)
#                 retriever = st.session_state.vectors.as_retriever()
#                 retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
#                 start_time = time.process_time()
#                 response = retrieval_chain.invoke({"input": prompt1})
#                 response_time = time.process_time() - start_time
                
#                 st.session_state.last_response = response.get('answer', "No answer found.")
#                 st.session_state.context_docs = response.get("context", [])
#                 st.session_state.response_time = f"{response_time:.2f}"
#                 st.session_state.translated_answer = None # Clear previous translation
#             except Exception as e:
#                 st.error(f"An error occurred while generating the answer: {e}")

# if translate and st.session_state.last_response:
#     with st.spinner("Translating to Urdu..."):
#         try:
#             translated = st.session_state.llm.invoke(f"Translate the following English text to Urdu: {st.session_state.last_response}")
#             st.session_state.translated_answer = translated.content
#         except Exception as e:
#             st.error(f"An error occurred during translation: {e}")

# # --- DISPLAYING RESULTS (adapted from new UI) ---
# if st.session_state.last_response:
#     st.markdown("---")
#     st.info(f"**Response time:** {st.session_state.response_time} seconds")
#     st.subheader("📌 Answer")
#     st.write(st.session_state.last_response)

#     # Display download buttons for source files
#     if st.session_state.context_docs:
#         st.markdown("---")
#         st.subheader("📚 Source References")
#         displayed_files = set()
#         for doc in st.session_state.context_docs:
#             filename = os.path.basename(doc.metadata.get("source", ""))
#             if filename and filename not in displayed_files:
#                 file_path = os.path.join("./static", filename)
#                 if os.path.exists(file_path):
#                     with open(file_path, "rb") as f:
#                         file_data = f.read()
#                     st.download_button(
#                         label=f"Download {filename}",
#                         data=file_data,
#                         file_name=filename,
#                         mime="application/pdf",
#                         key=f"download_{filename}"
#                     )
#                     displayed_files.add(filename)

# if st.session_state.translated_answer:
#     st.markdown("---")
#     st.success("Translation Complete!")
#     st.write("**🌍 Translated Answer (Urdu):**")
#     st.write(st.session_state.translated_answer)

# if st.session_state.context_docs:
#     with st.expander("📄 View Relevant Document Chunks"):
#         for i, doc in enumerate(st.session_state.context_docs):
#             st.markdown(f"**Chunk {i+1} from `{os.path.basename(doc.metadata.get('source', 'Unknown'))}`**")
#             st.write(doc.page_content)
#             st.markdown("---")


# ###############################################
# Adding some high performance features
# Running perfectly 

import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Page Configuration ---
st.set_page_config(page_title="PIA ChatBot", page_icon="⚖️", layout="centered")

# --- Load Environment Variables and API Keys ---
load_dotenv()
groq_api_key = os.environ.get('GROQ_API_KEY')
google_api_key = os.environ.get('GOOGLE_API_KEY')
if not groq_api_key or not google_api_key:
    st.error("API keys for Groq and Google are not set. Please check your .env file.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = google_api_key

# --- UI Styling ---
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stApp { background-color: #FFFFFF; }
        h1, h2, h3 { text-align: center; }
        .stButton>button { width: 100%; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- Main Title ---
st.markdown("""
    <h1 style='text-align: center;'>PIA Document Assistant</h1>
    <p style='text-align: center;'>Ask questions about your documents and get answers instantly.</p>
""", unsafe_allow_html=True)


# --- Initialize LLM and Session State ---
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

for key in ["last_response", "translated_answer", "response_time", "vectors", "context_docs"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Backend Functions ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def vector_embedding():
    FAISS_INDEX_PATH = "faiss_vector_store"
    if st.session_state.vectors is None:
        try:
            embeddings_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
            if os.path.exists(FAISS_INDEX_PATH):
                with st.spinner("Loading existing Vector Store..."):
                    st.session_state.vectors = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
                    st.success("Vector Store loaded successfully!")
            else:
                with st.spinner("Creating new Vector Store..."):
                    pdf_folder = "./static"
                    if not os.path.isdir(pdf_folder) or not os.listdir(pdf_folder):
                        st.sidebar.error(f"The '{pdf_folder}' directory is empty or does not exist.")
                        return
                    loader = PyPDFDirectoryLoader(pdf_folder)
                    docs = loader.load()
                    if not docs:
                        st.sidebar.warning("No PDF documents were found to process.")
                        return
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    final_documents = text_splitter.split_documents(docs)
                    st.session_state.vectors = FAISS.from_documents(final_documents, embeddings_model)
                    st.session_state.vectors.save_local(FAISS_INDEX_PATH)
                    st.success("New Vector Store created and saved successfully!")
        except Exception as e:
            st.sidebar.error(f"An error occurred: {e}")
    else:
        st.success("Vector Store is already loaded.")

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📁 Navigation")
    if st.button("📂 Create Vector Store"):
        vector_embedding()

    if st.session_state.vectors:
        st.sidebar.success("✅ Documents Loaded & Ready")

    st.markdown("---")
    st.markdown("### ℹ️ Instructions")
    st.info("""
    1. Place PDFs in the `./static` folder.
    2. Click the button above to process.
    3. Ask any question!
    """)

# --- Main Content Area ---
st.markdown("---")
st.markdown("### 📝 Ask Your Question")
prompt1 = st.text_area("🔎 Type your legal question below:", height=150, label_visibility="collapsed")

col1, col2 = st.columns(2)
with col1:
    submit = st.button("✅ Get Answer")
with col2:
    translate = st.button("🌐 Translate to Urdu")

# --- ### MODIFIED: Answering Logic with Intent Classifier ### ---
if submit and prompt1:
    st.session_state.last_response = None
    st.session_state.context_docs = None
    st.session_state.translated_answer = None

    if st.session_state.vectors is None:
        st.error("Please create the vector store first using the button in the sidebar.")
    else:
        # ### RE-INTRODUCED: Intent Classification Step ###
        # This is the "smarter" front gate to filter queries.
        intent_prompt = ChatPromptTemplate.from_template(
            "You are a query classifier. Your task is to determine if the user's input is a request for information from a document.\n"
            "- If the input is a question, a command to summarize, or a topic to find information about, classify it as 'Informational'.\n"
            "- For anything else, such as greetings, single vague words, or conversational chit-chat, classify it as 'Other'.\n"
            "Respond with only the category name: 'Informational' or 'Other'.\n\nUser Input: {user_input}"
        )
        intent_chain = intent_prompt | st.session_state.llm
        with st.spinner("Analyzing query..."):
            intent = intent_chain.invoke({"user_input": prompt1}).content.strip()

        # Route the logic based on the determined intent
        if intent == "Informational":
            prompt_template = ChatPromptTemplate.from_template(
                "Answer based only on the context provided.\n<context>{context}</context>\nQuestion: {input}"
            )
            with st.spinner("Searching documents and generating answer..."):
                try:
                    document_chain = create_stuff_documents_chain(st.session_state.llm, prompt_template)
                    retriever = st.session_state.vectors.as_retriever()
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    
                    start_time = time.process_time()
                    response = retrieval_chain.invoke({"input": prompt1})
                    response_time = time.process_time() - start_time
                    
                    st.session_state.last_response = response.get('answer', "No answer found.")
                    st.session_state.context_docs = response.get("context", [])
                    st.session_state.response_time = f"{response_time:.2f}"

                    # The Grounding Check remains as a final quality gate
                    if st.session_state.last_response and st.session_state.context_docs:
                        context_str = "\n\n".join([doc.page_content for doc in st.session_state.context_docs])
                        grounding_prompt = ChatPromptTemplate.from_template(
                            "You are a verifier... Is the 'Answer' supported by the 'Context'? Respond with only 'Yes' or 'No'.\n\n" # Prompt collapsed for brevity
                            "Context:\n---\n{context}\n---\n\n"
                            "Answer: {ai_answer}"
                        )
                        grounding_chain = grounding_prompt | st.session_state.llm
                        
                        with st.spinner("Verifying answer grounding..."):
                            is_grounded = grounding_chain.invoke({
                                "context": context_str, 
                                "ai_answer": st.session_state.last_response
                            }).content.strip()

                        if "No" in is_grounded:
                            st.session_state.context_docs = None

                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")
        
        else: # Handles 'Other' intents (greetings, keywords, etc.)
            st.session_state.last_response = "It seems you've entered a keyword or an incomplete phrase. Please ask a complete question about your documents."

# --- TRANSLATION LOGIC ---
if translate and st.session_state.last_response:
    with st.spinner("Translating to Urdu..."):
        try:
            translation_prompt = ChatPromptTemplate.from_template("Translate the following English text to Urdu: {text}")
            translation_chain = translation_prompt | st.session_state.llm
            translated_text = translation_chain.invoke({"text": st.session_state.last_response}).content
            st.session_state.translated_answer = translated_text
        except Exception as e:
            st.error(f"Translation failed: {e}")

# --- DISPLAYING RESULTS ---
if st.session_state.last_response:
    st.markdown("---")
    if st.session_state.response_time:
        st.info(f"**Response time:** {st.session_state.response_time} seconds")
    st.subheader("📌 Answer")
    st.write(st.session_state.last_response)

    if st.session_state.context_docs:
        st.markdown("---")
        st.subheader("📚 Source References")
        displayed_files = set()
        for doc in st.session_state.context_docs:
            filename = os.path.basename(doc.metadata.get("source", ""))
            if filename and filename not in displayed_files:
                file_path = os.path.join("./static", filename)
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        file_data = f.read()
                    st.download_button(
                        label=f"Download {filename}",
                        data=file_data,
                        file_name=filename,
                        mime="application/pdf",
                        key=f"download_{filename}_{time.time()}"
                    )
                    displayed_files.add(filename)
        
        with st.expander("📄 View Relevant Document Chunks"):
            for i, doc in enumerate(st.session_state.context_docs):
                st.markdown(f"**Chunk {i+1} from `{os.path.basename(doc.metadata.get('source', 'Unknown'))}`**")
                st.write(doc.page_content)
                st.markdown("---")

if st.session_state.translated_answer:
    st.markdown("---")
    st.success("Translation Complete!")
    st.write("**🌍 Translated Answer (Urdu):**")
    st.write(st.session_state.translated_answer)
            
            
# Major ui improvements

# import os
# import time
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_groq import ChatGroq
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from tenacity import retry, stop_after_attempt, wait_exponential

# # --- Page Configuration ---
# st.set_page_config(page_title="PIA ChatBot", page_icon="⚖️", layout="centered")

# # --- Load Environment Variables and API Keys ---
# load_dotenv()
# groq_api_key = os.environ.get('GROQ_API_KEY')
# google_api_key = os.environ.get('GOOGLE_API_KEY')
# if not groq_api_key or not google_api_key:
#     st.error("API keys for Groq and Google are not set. Please check your .env file.")
#     st.stop()
# os.environ["GOOGLE_API_KEY"] = google_api_key

# # --- UI Styling ---
# st.markdown("""
#     <style>
#         #MainMenu {visibility: hidden;}
#         footer {visibility: hidden;}
#         .stApp { background-color: #FFFFFF; }
#         h1, h2, h3 { text-align: center; }
#         .stButton>button { width: 100%; border-radius: 8px; }
#         .stChatMessage { border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
#     </style>
# """, unsafe_allow_html=True)

# # --- Initialize LLM and Session State ---
# if "llm" not in st.session_state:
#     st.session_state.llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
# if "vectors" not in st.session_state:
#     st.session_state.vectors = None
# if "messages" not in st.session_state:
#     st.session_state.messages = [{"role": "assistant", "content": "Hello! Load your documents using the sidebar and then ask me anything about them."}]


# # --- Backend Functions ---
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def vector_embedding():
#     FAISS_INDEX_PATH = "faiss_vector_store"
#     if st.session_state.vectors is None:
#         try:
#             embeddings_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
#             if os.path.exists(FAISS_INDEX_PATH):
#                 with st.spinner("Loading existing Vector Store..."):
#                     st.session_state.vectors = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
#             else:
#                 with st.spinner("Creating new Vector Store..."):
#                     pdf_folder = "./static"
#                     if not os.path.isdir(pdf_folder) or not os.listdir(pdf_folder):
#                         st.sidebar.error(f"The '{pdf_folder}' directory is empty or does not exist.")
#                         return
#                     loader = PyPDFDirectoryLoader(pdf_folder)
#                     docs = loader.load()
#                     if not docs:
#                         st.sidebar.warning("No PDF documents were found to process.")
#                         return
#                     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#                     final_documents = text_splitter.split_documents(docs)
#                     st.session_state.vectors = FAISS.from_documents(final_documents, embeddings_model)
#                     st.session_state.vectors.save_local(FAISS_INDEX_PATH)
#         except Exception as e:
#             st.sidebar.error(f"An error occurred: {e}")

# # --- Sidebar UI ---
# with st.sidebar:
#     st.header("Controls")
#     st.markdown("---")
#     if st.button("📂 Process & Load Documents"):
#         vector_embedding()
    
#     if st.session_state.vectors:
#         st.sidebar.success("✅ Documents Loaded & Ready")
    
#     st.markdown("---")
#     st.markdown("### ℹ️ Instructions")
#     st.info("1. Place PDFs in the `./static` folder.\n"
#             "2. Click the button above to process.\n"
#             "3. Ask any question!")

# # --- Main Chat Interface ---
# st.title("PIA Document Assistant")

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         if message["role"] == "assistant" and message.get("sources"):
#             with st.expander("📄 View Sources"):
#                 for doc in message["sources"]:
#                      st.markdown(f"**Chunk from `{os.path.basename(doc.metadata.get('source', 'Unknown'))}`**")
#                      st.write(doc.page_content)
#                      st.markdown("---")

# if prompt := st.chat_input("Ask a question about your documents..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             final_response_text = ""
#             context_docs = None

#             if st.session_state.vectors is None:
#                 final_response_text = "Please load the documents first using the button in the sidebar."
#             else:
#                 # ### UPDATED PROMPT ###
#                 # This new prompt is more flexible and gives examples of valid questions.
#                 intent_prompt = ChatPromptTemplate.from_template(
#                     "You are an expert at classifying user intent. Your goal is to determine if the user is asking for specific information from a document.\n"
#                     "- A 'Question' is any query asking for information, a summary, or details about a topic (e.g., 'what is the tender about?', 'summarize the report', 'details on the 2012 meeting').\n"
#                     "- A 'Greeting' is a simple social greeting (e.g., 'hello', 'hi').\n"
#                     "- A 'Keyword' is a query that is too vague or short to be answerable (e.g., 'the', 'is', 'a').\n"
#                     "Classify the following user input into 'Question', 'Greeting', or 'Keyword'. Respond with only the category name.\nUser Input: {user_input}"
#                 )
#                 intent_chain = intent_prompt | st.session_state.llm
#                 intent = intent_chain.invoke({"user_input": prompt}).content.strip()

#                 if intent == "Question":
#                     rag_prompt = ChatPromptTemplate.from_template(
#                         "Answer based only on the context provided.\n<context>{context}</context>\nQuestion: {input}"
#                     )
#                     document_chain = create_stuff_documents_chain(st.session_state.llm, rag_prompt)
#                     retriever = st.session_state.vectors.as_retriever()
#                     retrieval_chain = create_retrieval_chain(retriever, document_chain)
#                     response = retrieval_chain.invoke({"input": prompt})
#                     final_response_text = response.get('answer', "I couldn't generate an answer.")
#                     context_docs = response.get("context", [])
                    
#                     refusal_prompt = ChatPromptTemplate.from_template(
#                         "Is the following AI answer a refusal, a placeholder, or a request for more info (e.g., 'I don't know', 'Please ask a proper question')? Answer only 'Yes' or 'No'.\nAnswer: {ai_answer}"
#                     )
#                     refusal_chain = refusal_prompt | st.session_state.llm
#                     is_refusal = refusal_chain.invoke({"ai_answer": final_response_text}).content.strip()
#                     if "Yes" in is_refusal:
#                         context_docs = None

#                 elif intent == "Greeting":
#                     final_response_text = "Hello! How can I help you with your documents?"
#                 else: # Keyword or other
#                     final_response_text = "It seems you've entered a keyword or an incomplete phrase. Please ask a complete question."
            
#             st.markdown(final_response_text)
#             if context_docs:
#                 with st.expander("📄 View Sources"):
#                      for doc in context_docs:
#                          st.markdown(f"**Chunk from `{os.path.basename(doc.metadata.get('source', 'Unknown'))}`**")
#                          st.write(doc.page_content)
#                          st.markdown("---")
            
#             st.session_state.messages.append({"role": "assistant", "content": final_response_text, "sources": context_docs})