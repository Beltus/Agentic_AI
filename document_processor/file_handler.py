import os
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from io import BytesIO
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter

from langchain_text_splitters import MarkdownHeaderTextSplitter
from config import constants
from config.settings import settings
from utils.logging import logger


#class responsible for handling document parsing, caching, and chunking.

# The DocumentProcessor class ensures efficient document parsing and retrieval by leveraging:
# Docling for structured content extraction
# ChromaDB-compatible chunking for vector search
# A caching system to avoid redundant processing
class DocumentProcessor:

    #initialize document processor with 1) a predefined header structure for markdown-based chunking. 2) A cache directory for storing
    #document chunks 3) ensures cache directory exists.
    def __init__(self):
        self.headers = [("#", "Header 1"), ("##", "Header 2")]
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    #Purpose: ensures that the total size of uploaded files doesn't exceed a predefined limit
    #How it works: 1) Computes the total size of all uploaded files 2) Compares the total size against a fixed max total size
    #              3) Raises a 'ValueError' if the limit is exceeded.
    def validate_files(self, files: List) -> None:
        """Validate the total size of the uploaded files."""
        total_size = sum(f.size for f in files)
        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(f"Total size exceeds {constants.MAX_TOTAL_SIZE//1024//1024}MB limit")
    
    #Purpose: Handles the entire document processing pipeline, including caching and deduplication
    #How it works:
    # 1) Validates the uploaded files
    # 2) Generate a hash for each file's content to check if it has been processed before
    # 3) If cached, load data from cache. Else process the file using _process_file() method and store results in cache.
    # 4) Ensures that no duplicate chunks are stored across multiple files.
    # 5) Returns all file chunks

    def process(self, files: List) -> List:
        """Process files with caching for subsequent queries"""
        self.validate_files(files)
        all_chunks = []
        seen_hashes = set()
        
        for file in files:
            try:
                # Generate content-based hash for caching
                # with open(file.name, "rb") as f:
                #     file_hash = self._generate_hash(f.read())
                file_hash = self._generate_hash(file.read())
                
                cache_path = self.cache_dir / f"{file_hash}.pkl" #create path to cache path.
                
                #check if file is already cached and load from cache
                if self._is_cache_valid(cache_path):
                    logger.info(f"Loading from cache: {file.name}")
                    chunks = self._load_from_cache(cache_path)

                #if not cached, process the file and store in cache
                else:
                    logger.info(f"Processing and caching: {file.name}")
                    chunks = self._process_file(file) #split file into structured text chunks
                    self._save_to_cache(chunks, cache_path)
                
                # Deduplicate chunks across files
                for chunk in chunks:
                    chunk_hash = self._generate_hash(chunk.page_content.encode()) #generate unique hash per chunk
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)
                        
            except Exception as e:
                logger.error(f"Failed to process {file.name}: {str(e)}")
                continue
                
        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks

    #Purpose: Converts the documents into Markdown and splits it into structured text chunks
    def _process_file(self, file) -> List:
        """Original processing logic with Docling"""
        #skip unsupported files
        if not file.name.endswith(('.pdf', '.docx', '.txt', '.md')):
            logger.warning(f"Skipping unsupported file type: {file.name}")
            return []
        
        #Get the document bytes and wrap them in BytesIO "buffer"
        file_bytes = BytesIO(file.getvalue())

        #Create a Docling DocumentStream

        source = DocumentStream(name=file.name, stream=file_bytes)

        #uses Docling 'DocumentConverter' to convert file to Markdown
        converter = DocumentConverter()
        result = converter.convert(source)
        markdown = result.document.export_to_markdown()

        #Split extracted Markdown text into chunks
        splitter = MarkdownHeaderTextSplitter(self.headers)
        return splitter.split_text(markdown)

    #Purpose: Generate a unique SHA-256 hash from doucment content
    def _generate_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    #Save processed document chunks in a pickle file for future use
    def _save_to_cache(self, chunks: List, cache_path: Path):
        
        #store chunks with timestanp for expiration checking
        with open(cache_path, "wb") as f:
            pickle.dump({
                "timestamp": datetime.now().timestamp(),
                "chunks": chunks
            }, f)

    #Purpose: Load cached document chunks from a previously processed file
    def _load_from_cache(self, cache_path: Path) -> List:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"]

    #Purpose: Check if cached file is still valid (not expired)
    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        
        #compare modification timestamp of cached file against CACHE_EXPIRE_DAYS
        #if file is older than expiration threshold, it is considered invalid.
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=settings.CACHE_EXPIRE_DAYS)