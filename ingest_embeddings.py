"""Embedding generation for document chunks.

Supports both local and API-based embedding models.
"""
import logging
from typing import Optional, List

import httpx
from openai import AsyncOpenAI


class EmbeddingGenerator:
    """Generates embeddings for text chunks."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        dimension: int = 768
    ):
        """Initialize embedding generator.
        
        Args:
            provider: Provider name (openai, ollama, etc.)
            model: Model name
            base_url: Base URL for API
            api_key: API key (if required)
            dimension: Embedding dimension
        """
        self.provider = provider
        self.model = model
        self.dimension = dimension
        
        if base_url:
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key or "sk-no-key-required"
            )
        else:
            self.client = AsyncOpenAI(api_key=api_key)
            
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            # Truncate text if too long (most models have token limits)
            max_chars = 8000  # Conservative limit
            if len(text) > max_chars:
                text = text[:max_chars]
                logging.warning(f"Truncated text to {max_chars} chars for embedding")
                
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Verify dimension
            if len(embedding) != self.dimension:
                logging.warning(
                    f"Expected dimension {self.dimension} but got {len(embedding)}"
                )
                
            return embedding
            
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            raise
            
    async def generate_embeddings_batch(
        self, 
        texts: List[str],
        batch_size: int = 10
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to embed in each API call
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Truncate texts
                max_chars = 8000
                truncated_batch = [
                    text[:max_chars] if len(text) > max_chars else text
                    for text in batch
                ]
                
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=truncated_batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logging.error(f"Error generating batch embeddings: {e}")
                # Return None for failed batches
                embeddings.extend([None] * len(batch))
                
        return embeddings
        
    async def close(self) -> None:
        """Close the HTTP client."""
        if hasattr(self.client, '_client') and hasattr(self.client._client, 'aclose'):
            await self.client._client.aclose()


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count.
    
    Uses a simple heuristic: ~4 characters per token for English text.
    This is not accurate but provides a reasonable approximation.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Simple heuristic: average 4 chars per token
    return len(text) // 4
