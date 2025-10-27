"""Content extraction utilities for various sources.

Handles HTML to Markdown conversion, Twitter/X scraping, arXiv processing, and file attachments.
"""
import logging
import re
from typing import Optional, Tuple
from urllib.parse import urlparse

import httpx

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import html2text
except ImportError:
    html2text = None


class ContentExtractor:
    """Extracts and normalizes content from various sources."""
    
    def __init__(self, httpx_client: httpx.AsyncClient):
        self.httpx_client = httpx_client
        
    async def extract_from_url(self, url: str) -> Tuple[bytes, str, str, str]:
        """Extract content from a URL.
        
        Returns:
            Tuple of (raw_content, markdown_text, plain_text, title)
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Special handling for specific domains
        # Use exact domain matching or proper suffix matching to avoid bypasses
        if domain == 'twitter.com' or domain.endswith('.twitter.com') or domain == 'x.com' or domain.endswith('.x.com'):
            return await self._extract_twitter(url)
        elif domain == 'arxiv.org' or domain.endswith('.arxiv.org'):
            return await self._extract_arxiv(url)
        else:
            return await self._extract_generic_webpage(url)
            
    async def _extract_twitter(self, url: str) -> Tuple[bytes, str, str, str]:
        """Extract content from Twitter/X posts and threads.
        
        Note: This is a basic implementation. For production, consider using
        Twitter API or specialized scraping tools.
        """
        try:
            # Try to get the page
            response = await self.httpx_client.get(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                },
                follow_redirects=True
            )
            response.raise_for_status()
            
            raw_content = response.content
            html = response.text
            
            if not BeautifulSoup:
                # Fallback: simple text extraction
                plain_text = html
                markdown_text = html
                title = "Twitter/X Post"
            else:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Try to extract tweet content
                # Note: Twitter/X HTML structure changes frequently
                # This is a basic extraction that may need updates
                title = "Twitter/X Post"
                
                # Look for article or main content
                article = soup.find('article')
                if article:
                    # Extract text from tweet
                    tweet_texts = []
                    for elem in article.find_all(['div', 'span'], recursive=True):
                        text = elem.get_text(strip=True)
                        if text and len(text) > 20:  # Filter short strings
                            tweet_texts.append(text)
                    
                    plain_text = '\n\n'.join(tweet_texts[:10])  # Limit to avoid duplicates
                else:
                    plain_text = soup.get_text(separator='\n', strip=True)
                
                # Try to get title from meta tags
                og_title = soup.find('meta', property='og:title')
                if og_title and og_title.get('content'):
                    title = og_title.get('content')
                    
                markdown_text = f"# {title}\n\n{plain_text}"
                
            return raw_content, markdown_text, plain_text, title
            
        except Exception as e:
            logging.error(f"Error extracting Twitter content: {e}")
            raise
            
    async def _extract_arxiv(self, url: str) -> Tuple[bytes, str, str, str]:
        """Extract content from arXiv papers.
        
        Handles both abstract pages and PDF links.
        """
        try:
            # Normalize arXiv URL to abstract page
            arxiv_id = None
            if 'abs/' in url:
                arxiv_id = url.split('abs/')[-1].split('?')[0]
            elif 'pdf/' in url:
                arxiv_id = url.split('pdf/')[-1].replace('.pdf', '').split('?')[0]
            elif match := re.search(r'(\d{4}\.\d{4,5})', url):
                arxiv_id = match.group(1)
                
            if not arxiv_id:
                raise ValueError(f"Could not extract arXiv ID from {url}")
                
            # Get abstract page
            abstract_url = f"https://arxiv.org/abs/{arxiv_id}"
            response = await self.httpx_client.get(abstract_url, follow_redirects=True)
            response.raise_for_status()
            
            raw_content = response.content
            html = response.text
            
            if not BeautifulSoup:
                plain_text = html
                markdown_text = html
                title = f"arXiv:{arxiv_id}"
            else:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract title
                title_elem = soup.find('h1', class_='title')
                if title_elem:
                    title = title_elem.get_text(strip=True).replace('Title:', '').strip()
                else:
                    title = f"arXiv:{arxiv_id}"
                    
                # Extract authors
                authors_elem = soup.find('div', class_='authors')
                authors = ""
                if authors_elem:
                    authors = authors_elem.get_text(strip=True)
                    
                # Extract abstract
                abstract_elem = soup.find('blockquote', class_='abstract')
                abstract = ""
                if abstract_elem:
                    abstract = abstract_elem.get_text(strip=True).replace('Abstract:', '').strip()
                    
                # Extract other metadata
                comments_elem = soup.find('td', class_='tablecell comments')
                comments = ""
                if comments_elem:
                    comments = comments_elem.get_text(strip=True)
                    
                # Build markdown
                markdown_parts = [f"# {title}\n"]
                if authors:
                    markdown_parts.append(f"**{authors}**\n")
                markdown_parts.append(f"**arXiv ID:** {arxiv_id}\n")
                markdown_parts.append(f"**URL:** {abstract_url}\n")
                if comments:
                    markdown_parts.append(f"**Comments:** {comments}\n")
                markdown_parts.append(f"\n## Abstract\n\n{abstract}")
                
                markdown_text = '\n'.join(markdown_parts)
                plain_text = f"{title}\n\n{authors}\n\n{abstract}"
                
            return raw_content, markdown_text, plain_text, title
            
        except Exception as e:
            logging.error(f"Error extracting arXiv content: {e}")
            raise
            
    async def _extract_generic_webpage(self, url: str) -> Tuple[bytes, str, str, str]:
        """Extract content from a generic webpage."""
        try:
            response = await self.httpx_client.get(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                },
                follow_redirects=True
            )
            response.raise_for_status()
            
            raw_content = response.content
            html = response.text
            
            if not BeautifulSoup:
                # Fallback: simple text extraction
                plain_text = html
                markdown_text = html
                title = url
            else:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract title
                title_elem = soup.find('title')
                if title_elem:
                    title = title_elem.get_text(strip=True)
                else:
                    # Try Open Graph title
                    og_title = soup.find('meta', property='og:title')
                    if og_title and og_title.get('content'):
                        title = og_title.get('content')
                    else:
                        title = url
                        
                # Remove script and style elements
                for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                    script.decompose()
                    
                # Try to find main content
                main_content = soup.find('main') or soup.find('article') or soup.find('body')
                
                if html2text:
                    # Convert to markdown using html2text
                    h = html2text.HTML2Text()
                    h.ignore_links = False
                    h.ignore_images = False
                    h.ignore_emphasis = False
                    h.body_width = 0  # Don't wrap lines
                    
                    if main_content:
                        markdown_text = h.handle(str(main_content))
                    else:
                        markdown_text = h.handle(html)
                        
                    # Extract plain text from markdown
                    plain_text = main_content.get_text(separator='\n', strip=True) if main_content else soup.get_text(separator='\n', strip=True)
                else:
                    # Fallback: just extract text
                    plain_text = main_content.get_text(separator='\n', strip=True) if main_content else soup.get_text(separator='\n', strip=True)
                    markdown_text = f"# {title}\n\n{plain_text}"
                    
            return raw_content, markdown_text, plain_text, title
            
        except Exception as e:
            logging.error(f"Error extracting webpage content: {e}")
            raise
            
    async def extract_from_attachment(
        self, 
        content: bytes, 
        filename: str, 
        content_type: str
    ) -> Tuple[bytes, str, str, str]:
        """Extract content from a file attachment.
        
        Returns:
            Tuple of (raw_content, markdown_text, plain_text, title)
        """
        title = filename
        
        # Text files
        if content_type.startswith('text/'):
            try:
                text = content.decode('utf-8')
                markdown_text = f"# {filename}\n\n```\n{text}\n```"
                return content, markdown_text, text, title
            except UnicodeDecodeError:
                pass
                
        # PDF files
        if content_type == 'application/pdf':
            # For PDFs, we would need a PDF parser like PyPDF2 or pdfplumber
            # For now, just store the raw content
            markdown_text = f"# {filename}\n\nPDF file (binary content)"
            plain_text = f"PDF file: {filename}"
            return content, markdown_text, plain_text, title
            
        # Other file types - store as-is
        markdown_text = f"# {filename}\n\nBinary file ({content_type})"
        plain_text = f"File: {filename} ({content_type})"
        return content, markdown_text, plain_text, title


def create_chunks(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 120
) -> list[Tuple[int, int, str]]:
    """Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters (approximates tokens)
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of (start_pos, end_pos, chunk_text) tuples
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundaries
        if end < len(text):
            # Look for sentence ending punctuation
            for i in range(end, max(start + chunk_size // 2, end - 200), -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break
                    
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append((start, end, chunk_text))
            
        # Move start position with overlap
        start = end - chunk_overlap
        
        # Prevent infinite loop
        if start >= len(text):
            break
            
    return chunks
