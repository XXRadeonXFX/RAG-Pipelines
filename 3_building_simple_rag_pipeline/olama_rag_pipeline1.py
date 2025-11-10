import os
import sys

# Fix OpenMP conflict on macOS - MUST BE BEFORE OTHER IMPORTS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
# LangChain recommended imports for Ollama
from langchain_community.llms import Ollama 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import numpy as np
import tempfile
from langchain_community.document_loaders import BSHTMLLoader
import speech_recognition as sr  # For voice input
import subprocess  # For macOS TTS

# Configuration variables
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "llama3.2"
TEMPERATURE = 0.4
ENABLE_SPEECH_OUTPUT = True  # Voice output
ENABLE_VOICE_INPUT = True    # Voice input (Siri-like)
VOICE_NAME = "Samantha"      # macOS voice (Samantha, Alex, Victoria, Daniel)

print("="*60)
print("🎤 SIRI-LIKE RAG ASSISTANT")
print("="*60)
print(f"Using Ollama Model: {MODEL_NAME}")
print(f"Voice Output: {'ON' if ENABLE_SPEECH_OUTPUT else 'OFF'}")
print(f"Voice Input: {'ON' if ENABLE_VOICE_INPUT else 'OFF'}")
print(f"Voice: {VOICE_NAME}")
print("="*60)
print()

# Initialize speech recognizer
if ENABLE_VOICE_INPUT:
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    # 🌟 FIX 1: Set a reliable energy threshold to filter noise on M1 Mac
    # Default is 300. 800-1000 is a good starting point for clearer audio environments.
    recognizer.energy_threshold = 800  
    
    print("✓ Speech Recognition initialized")
else:
    recognizer = None
    microphone = None

def speak_text(text):
    """Convert text to speech using macOS say command (more natural than pyttsx3)"""
    if ENABLE_SPEECH_OUTPUT:
        try:
            print("🔊 Speaking response...")
            # Using macOS 'say' command for better quality
            subprocess.run(['say', '-v', VOICE_NAME, text], check=True)
        except Exception as e:
            print(f"Error in speech output: {e}")

def listen_for_command():
    """Listen for voice input and convert to text"""
    if not ENABLE_VOICE_INPUT or not recognizer or not microphone:
        return None
    
    # FIX 2: Removed speech output to prevent audio corruption during recording
    print("\n🎤 Listening... (Speak now)")

    try:
        with microphone as source:
            # FIX 3: Increased duration for more robust ambient noise adjustment on M1 Mac
            recognizer.adjust_for_ambient_noise(source, duration=1.5)
            # Listen for audio
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        print("🔄 Processing speech...")
        # Speak confirmation after recording, but before the internet API call
        speak_text("Got it.") 
        
        # Convert speech to text using Google Speech Recognition
        # NOTE: If this fails, you MUST use a dedicated Google Cloud API Key here
        # Example: text = recognizer.recognize_google(audio, key="YOUR_API_KEY")
        text = recognizer.recognize_google(audio)
        
        print(f"📝 You said: {text}")
        return text
    
    except sr.WaitTimeoutError:
        print("⏱️  No speech detected. Please try again.")
        speak_text("I didn't hear anything. Please try again.")
        return None
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        speak_text("Sorry, I couldn't understand that. Please speak clearly.")
        return None
    except sr.RequestError as e:
        # This error is usually a temporary server issue or API limit failure
        print(f"❌ Could not request results; recognition request failed: {e}")
        speak_text("Sorry, I'm having trouble with speech recognition. Please try again or use text input.")
        return None
    except Exception as e:
        print(f"❌ General Error: {e}")
        return None

def get_user_input(prompt_text):
    """Get input either by voice or text"""
    if ENABLE_VOICE_INPUT:
        print(f"\n{prompt_text}")
        print("Choose: [V]oice or [T]ext input? (or type your question directly)")
        
        choice = input(">>> ").strip().lower()
        
        if choice == 'v' or choice == 'voice':
            return listen_for_command()
        elif choice == 't' or choice == 'text':
            return input("Type your query: ").strip()
        else:
            # User typed directly
            return choice
    else:
        return input(f"{prompt_text}: ").strip()

def fetch_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching the website: {e}")
        return None

def process_website(url):
    html_content = fetch_html(url)
    if not html_content:
        raise ValueError("No content could be fetched from the website.")
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as temp_file:
        temp_file.write(html_content)
        temp_file_path = temp_file.name

    try:
        try:
            loader = BSHTMLLoader(temp_file_path)
        except ImportError:
            print("'lxml' not installed. Using 'html.parser'.")
            loader = BSHTMLLoader(temp_file_path, bs_kwargs={'features': 'html.parser'})
        documents = loader.load()
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

    print(f"\nNumber of documents loaded: {len(documents)}")
    if documents:
        print("Sample of loaded content:")
        print(documents[0].page_content[:200] + "...")
    
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    print(f"Number of text chunks after splitting: {len(texts)}")
    return texts

def print_sample_embeddings(texts, embeddings):
    if texts:
        sample_text = texts[0].page_content
        print("\nGenerating sample embedding...")
        sample_embedding = embeddings.embed_query(sample_text)
        print(f"\nEmbedding shape: {np.array(sample_embedding).shape}")

# Set up Ollama language model
try:
    llm = Ollama(model=MODEL_NAME, temperature=TEMPERATURE)
    print("✓ Ollama LLM initialized successfully")
except Exception as e:
    print(f"Error initializing Ollama: {e}")
    print("Make sure Ollama is running: ollama serve")
    exit(1)

template = """Context: {context}

Question: {question}

Answer the question concisely based only on the given context. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."

But, if the question is generic, then go ahead and answer the question, example what is a electric vehicle?
"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

def rag_pipeline(query, qa_chain, vectorstore):
    relevant_docs = vectorstore.similarity_search_with_score(query, k=3)
    
    print("\nTop 3 most relevant chunks:")
    for i, (doc, score) in enumerate(relevant_docs, 1):
        print(f"{i}. Relevance Score: {score:.4f}")
        print(f"   Content: {doc.page_content[:100]}...")
        print()

    print("🤔 Thinking... (this may take 10-30 seconds)")
    response = qa_chain.invoke({"query": query})
    return response['result']

if __name__ == "__main__":
    print("\n🚀 Welcome to the Siri-Like RAG Assistant!")
    speak_text("Hello! I'm your voice assistant. Let's get started.")
    
    while True:
        url_input = get_user_input("\nEnter website URL (or say/type 'quit' to exit')")
        
        if not url_input or url_input.lower() == 'quit':
            speak_text("Goodbye!")
            print("Exiting. Goodbye!")
            break
        
        try:
            print("\n📥 Processing website content...")
            speak_text("Processing the website. This will take a moment.")
            
            texts = process_website(url_input)
            
            if not texts:
                print("No content found on the website.")
                speak_text("Sorry, I couldn't find any content on that website.")
                continue
            
            print("\n🧠 Creating embeddings with Ollama...")
            speak_text("Creating knowledge base from the website.")
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            print_sample_embeddings(texts, embeddings)
            
            print("\n💾 Building vector store...")
            vectorstore = FAISS.from_documents(texts, embeddings)
            
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            print("\n" + "="*60)
            print("✅ RAG Pipeline Ready!")
            print("You can now ask questions using voice or text")
            print("Say/type 'new' for new website, 'quit' to exit")
            print("="*60)
            speak_text("I'm ready! What would you like to know?")
            
            while True:
                user_query = get_user_input("\n🎤 Ask your question")
                
                if not user_query:
                    continue
                    
                if user_query.lower() == 'quit':
                    speak_text("Goodbye!")
                    print("Exiting. Goodbye!")
                    exit() 
                elif user_query.lower() == 'new':
                    speak_text("Okay, let's look at a new website.")
                    break
                
                result = rag_pipeline(user_query, qa, vectorstore)
                
                print(f"\n{'='*60}")
                print(f"💬 Response:\n{result}")
                print(f"{'='*60}")
                
                # Speak the response
                speak_text(result)
        
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            speak_text("Sorry, I encountered an error. Please try again.")