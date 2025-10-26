# Document Ingestion Feature

The `/ingest` command allows you to ingest links and file attachments from Discord channels into a searchable knowledge base.

## Features

- **Dual storage backends**: SQLite + FTS5 + FAISS or Postgres + pgvector
- **Content extraction**: Automatic HTML to Markdown conversion
- **Special handling**: Twitter/X posts and arXiv papers
- **File support**: Text files, PDFs, and other attachments
- **Chunking**: Automatic text chunking with overlaps
- **Embeddings**: Vector embeddings for semantic search
- **Hybrid search**: Combines lexical (FTS) and vector search
- **Provenance**: Tracks source Discord messages

## Setup

### 1. Install Dependencies

For SQLite backend (recommended for self-hosted):
```bash
pip install aiosqlite beautifulsoup4 html2text faiss-cpu numpy
```

For Postgres backend (for Supabase or production):
```bash
pip install asyncpg beautifulsoup4 html2text
```

### 2. Configure

Add to your `config.yaml`:

```yaml
ingest:
  enabled: true
  
  # Storage backend: 'sqlite' or 'postgres'
  backend: sqlite
  
  # Connection string
  # SQLite: path to database file
  connection_string: ingest.db
  
  # For Postgres:
  # connection_string: postgres://user:pass@host:port/dbname
  
  # Embedding settings
  embedding:
    provider: openai
    model: text-embedding-3-small
    dimension: 768
    # Optional: override base_url and api_key
    # base_url: http://localhost:11434/v1
  
  # Chunking parameters
  chunk_size: 900        # characters (~225 tokens)
  chunk_overlap: 120     # overlap between chunks
```

### 3. Set Admin Permissions

Only admins can use the `/ingest` command. Add your Discord user ID to the admin list:

```yaml
permissions:
  users:
    admin_ids: [YOUR_USER_ID_HERE]
```

## Usage

1. **Run the `/ingest` command** in any Discord channel
2. The bot will scan the last 100 messages in that channel
3. It will extract:
   - All URLs (webpages, Twitter/X posts, arXiv papers, etc.)
   - All file attachments (text files, PDFs, etc.)
4. For each item:
   - Downloads and extracts content
   - Converts to normalized Markdown
   - Chunks the text
   - Generates embeddings
   - Stores in the database with full provenance

## Supported Content

### URLs
- **Generic webpages**: Extracts title and main content
- **Twitter/X posts**: Extracts tweet text and metadata
- **arXiv papers**: Extracts title, authors, abstract, and metadata
- Any other HTTP(S) URL

### File Attachments
- **Text files**: `.txt`, `.py`, `.js`, `.md`, etc.
- **PDFs**: Binary storage (text extraction requires additional libraries)
- **Other files**: Binary storage with metadata

## Storage Schema

### SQLite Backend

Tables:
- `source_messages`: Discord message provenance
- `documents`: Document metadata and content hashes
- `document_versions`: Normalized text (Markdown and plain)
- `document_chunks`: Text chunks with embeddings
- `document_chunks_fts`: FTS5 index for lexical search
- `collections`: Grouping documents (e.g., by channel)
- `ingest_jobs`: Ingestion job tracking

FAISS index is built in memory for vector search.

### Postgres Backend

Uses the schema described in the problem statement with:
- pgvector extension for vector search
- pg_trgm for trigram matching
- tsvector for full-text search
- IVFFlat or HNSW indexes

## Search (Hybrid)

The system supports hybrid search combining:
- **Lexical search**: FTS5 (SQLite) or tsvector (Postgres)
- **Vector search**: FAISS (SQLite) or pgvector (Postgres)
- **Fusion**: Weighted combination (60% vector + 40% lexical)

To use search, you can extend the code to add a `/search` command using `IngestDB.search_hybrid()`.

## Deduplication

Documents are deduplicated by SHA-256 hash. The same content will not be ingested twice.

## Notes

- **Network access**: The bot needs internet access to fetch URLs
- **Rate limits**: Be mindful of API rate limits for embedding generation
- **Storage**: SQLite is simpler for self-hosted; Postgres scales better
- **Supabase**: For Supabase, use Postgres backend and configure Supabase Storage for raw files

## Example Output

```
📥 Ingesting 15 items... (15/15)
✅ 12 ingested | ❌ 2 failed | ⏭️ 1 skipped

✅ Ingestion complete!

Results:
• 12 documents ingested
• 2 failed
• 1 skipped (no content)
• Total items processed: 15
```

## Troubleshooting

**"Ingestion feature not available"**: Install the required dependencies

**"Ingestion is not enabled"**: Set `ingest.enabled: true` in config.yaml

**"You don't have permission"**: Add your user ID to `admin_ids` in config

**Failed to extract content**: Some websites block bots or require authentication

**Embedding generation failed**: Check your embedding provider API key and rate limits
