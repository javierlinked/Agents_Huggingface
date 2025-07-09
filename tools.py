import os
import tempfile
import requests
from google import genai
from google.genai import types
from langchain_core.tools import tool
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_tavily import TavilySearch

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a - b

@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers.
    
    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a % b

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.
    
    Args:
        query: The search query.
    Returns:
        str: Formatted search results from Wikipedia or an error message if no results are found.
    """
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    
    if not search_docs:
        return f"No Wikipedia articles found for query: {query}"
    
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata.get("source", "")}" page="{doc.metadata.get("title", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return formatted_search_docs

@tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return maximum 3 results.
    
    Args:
        query: The search query.

    Returns:
        str: Formatted search results from Tavily or an error message if no results are found.   
    """
    search_docs = TavilySearch(max_results=3).invoke(query)
    
    if not search_docs:
        return f"No web search results found for query: {query}"
    
    # Note: TavilySearch returns dictionaries, not Document objects
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.get("url", "")}" />\n{doc.get("content", "")}\n</Document>'
            for doc in search_docs
        ])
    return formatted_search_docs

@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    
    if not search_docs:
        return f"No arXiv articles found for query: {query}"
    
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata.get("source", "")}" page="{doc.metadata.get("title", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return formatted_search_docs

@tool
def youtube_analyze(question: str, url: str):
    """Given a question and YouTube URL, analyze the video to answer the question.

        Args:
            question (str): Question about a YouTube video
            url (str): The YouTube URL
            
        Returns:
            str: Answer to the question about the YouTube video
            
        Raises:
            RuntimeError: If processing fails"""
    try:
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

        return client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=types.Content(
                parts=[types.Part(file_data=types.FileData(file_uri=url)),
                        types.Part(text=question)]
            )
        )
    except Exception as e:
        raise RuntimeError(f"Processing failed: {str(e)}")
        
@tool
def reverse_string(s: str) -> str:
    """Reverse a given string.
    
    Args:
        s: The string to reverse."""
    return s[::-1]

@tool
def analyze_audio_file(path_file_audio: str, query: str):
    """
    Analyzes an MP3 audio file to answer a specific query.
    Args:
        path_file_audio (str): Path to the MP3 audio file to be analyzed.
        query (str): Question or query to analyze the content of the audio file.
    Returns:
        str: The result of the analysis of audio.
    Raises:
        Exception: If there is an error during the analysis of the audio file.
    """
    print(f"Analyze audio file tool. Analyzing audio file: {path_file_audio} with query: {query}")
    try:
        client = genai.Client(api_key=os.getenv("Gemini_API_KEY"))

        myfile = client.files.upload(file=path_file_audio)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[f"Carefully analyze the audio to answer the question correctly.\n\n The question is {query}",
                    myfile]
        )

        return response.text
    except Exception as e:
        print(f"Error analyzing audio file: {e!s}")
        return f"Error analyzing audio file: {e!s}"

@tool
def download_file_of_task_id(task_id: str, file_name: str) -> str:
    """
    Download a file associated with a specific task ID and save it to a temporary location.
    Args:
        task_id (str): The unique identifier of the task associated with the file to download.
        file_name (str): The name to assign to the downloaded file.
    Returns:
        str: Path to the downloaded file or an error message if the download fails.
    Raises:
        Exception: If there is an error during the download process.
    """
    print(f"Download file tool. Downloading file for task_id: {task_id} with file_name: {file_name}")
    try:
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, file_name)
        get_file_url = f"https://agents-course-unit4-scoring.hf.space/files/{task_id}"

        print(f"get file url for task_id: {task_id} and file_name: {file_name}: {get_file_url}")

        # Download the file
        response = requests.get(get_file_url, stream=True)
        response.raise_for_status()

        # Save the file
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return filepath
    except Exception as e:
        print(f"Error downloading file: {e!s}")
        return f"Error downloading file: {e!s}"


def get_tools():
    """Get all tools."""
    return [
        multiply,
        add,
        subtract,
        divide,
        modulus,
        wiki_search,
        web_search,
        arvix_search,
        youtube_analyze,
        reverse_string,
        analyze_audio_file,
        download_file_of_task_id
    ]