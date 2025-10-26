"""Database schemas and operations for document ingestion.

Supports both SQLite+FTS5+FAISS and Postgres+pgvector backends.
"""
import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import uuid

try:
    import aiosqlite
except ImportError:
    aiosqlite = None

try:
    import asyncpg
except ImportError:
    asyncpg = None

try:
    import numpy as np
    import faiss
except ImportError:
    faiss = None
    np = None


class StorageBackend(Enum):
    """Available storage backends."""
    SQLITE = "sqlite"
    POSTGRES = "postgres"


class ContentKind(Enum):
    """Types of content that can be ingested."""
    WEBPAGE = "webpage"
    PDF = "pdf"
    DOC = "doc"
    SHEET = "sheet"
    SLIDE = "slide"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"
    OTHER = "other"


class IngestStatus(Enum):
    """Status of an ingestion job."""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class DocumentChunk:
    """Represents a chunk of text with embedding."""
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    embedding: Optional[list[float]] = None
    token_count: Optional[int] = None
    meta: dict[str, Any] = None


@dataclass
class Document:
    """Represents an ingested document."""
    id: str
    source_message_id: Optional[str]
    url: Optional[str]
    sha256_hex: str
    title: Optional[str]
    content_kind: ContentKind
    text_md: Optional[str]
    text_plain: Optional[str]
    chunks: list[DocumentChunk]
    meta: dict[str, Any]
    created_at: datetime


class IngestDB:
    """Database interface for document ingestion."""
    
    def __init__(self, backend: StorageBackend, connection_string: str):
        self.backend = backend
        self.connection_string = connection_string
        self._conn = None
        self._faiss_index = None
        self._chunk_id_map = {}  # Maps FAISS index -> chunk_id
        
    async def connect(self) -> None:
        """Connect to the database."""
        if self.backend == StorageBackend.SQLITE:
            if aiosqlite is None:
                raise RuntimeError("aiosqlite not installed. Install with: pip install aiosqlite")
            self._conn = await aiosqlite.connect(self.connection_string)
            await self._init_sqlite_schema()
        elif self.backend == StorageBackend.POSTGRES:
            if asyncpg is None:
                raise RuntimeError("asyncpg not installed. Install with: pip install asyncpg")
            self._conn = await asyncpg.connect(self.connection_string)
            await self._init_postgres_schema()
            
    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            if self.backend == StorageBackend.SQLITE:
                await self._conn.close()
            elif self.backend == StorageBackend.POSTGRES:
                await self._conn.close()
            self._conn = None
            
    async def _init_sqlite_schema(self) -> None:
        """Initialize SQLite schema with FTS5."""
        await self._conn.executescript("""
            -- Source messages (Discord provenance)
            CREATE TABLE IF NOT EXISTS source_messages (
                id TEXT PRIMARY KEY,
                platform TEXT NOT NULL DEFAULT 'discord',
                guild_id TEXT,
                channel_id TEXT,
                message_id TEXT UNIQUE,
                author_id TEXT,
                posted_at TEXT,
                raw_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Documents
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                source_message_id TEXT REFERENCES source_messages(id) ON DELETE SET NULL,
                url TEXT,
                storage_bucket TEXT,
                storage_path TEXT,
                sha256_hex TEXT UNIQUE,
                title TEXT,
                author TEXT,
                description TEXT,
                content_kind TEXT NOT NULL,
                bytes INTEGER,
                lang TEXT,
                meta TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS documents_url_idx ON documents(url);
            
            -- Document versions
            CREATE TABLE IF NOT EXISTS document_versions (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                version INTEGER NOT NULL,
                normalized_format TEXT NOT NULL DEFAULT 'markdown',
                text_len INTEGER NOT NULL,
                text_md TEXT,
                text_plain TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(document_id, version)
            );
            
            -- Document chunks
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                document_version_id TEXT NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                char_start INTEGER NOT NULL,
                char_end INTEGER NOT NULL,
                token_count INTEGER,
                text TEXT NOT NULL,
                embedding_json TEXT,
                meta TEXT NOT NULL DEFAULT '{}',
                UNIQUE(document_version_id, chunk_index)
            );
            
            -- FTS5 virtual table for full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
                chunk_id UNINDEXED,
                text,
                content=document_chunks,
                content_rowid=rowid
            );
            
            -- Triggers to keep FTS5 in sync
            CREATE TRIGGER IF NOT EXISTS document_chunks_fts_insert AFTER INSERT ON document_chunks BEGIN
                INSERT INTO document_chunks_fts(rowid, chunk_id, text)
                VALUES (new.rowid, new.id, new.text);
            END;
            
            CREATE TRIGGER IF NOT EXISTS document_chunks_fts_delete AFTER DELETE ON document_chunks BEGIN
                DELETE FROM document_chunks_fts WHERE rowid = old.rowid;
            END;
            
            CREATE TRIGGER IF NOT EXISTS document_chunks_fts_update AFTER UPDATE ON document_chunks BEGIN
                DELETE FROM document_chunks_fts WHERE rowid = old.rowid;
                INSERT INTO document_chunks_fts(rowid, chunk_id, text)
                VALUES (new.rowid, new.id, new.text);
            END;
            
            -- Collections
            CREATE TABLE IF NOT EXISTS collections (
                id TEXT PRIMARY KEY,
                slug TEXT UNIQUE NOT NULL,
                title TEXT,
                meta TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Collection membership
            CREATE TABLE IF NOT EXISTS collection_membership (
                collection_id TEXT REFERENCES collections(id) ON DELETE CASCADE,
                document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
                PRIMARY KEY (collection_id, document_id)
            );
            
            -- Ingestion jobs
            CREATE TABLE IF NOT EXISTS ingest_jobs (
                id TEXT PRIMARY KEY,
                source_message_id TEXT REFERENCES source_messages(id) ON DELETE SET NULL,
                document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
                url TEXT,
                storage_path TEXT,
                status TEXT NOT NULL DEFAULT 'queued',
                error TEXT,
                timings_ms TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await self._conn.commit()
        logging.info("SQLite schema initialized")
        
    async def _init_postgres_schema(self) -> None:
        """Initialize Postgres schema with pgvector."""
        await self._conn.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE EXTENSION IF NOT EXISTS pg_trgm;
            CREATE EXTENSION IF NOT EXISTS unaccent;
            CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        """)
        
        # Create enums
        await self._conn.execute("""
            DO $$ BEGIN
                CREATE TYPE content_kind AS ENUM (
                    'webpage', 'pdf', 'doc', 'sheet', 'slide', 'image', 'audio', 'video', 'code', 'other'
                );
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """)
        
        await self._conn.execute("""
            DO $$ BEGIN
                CREATE TYPE ingest_status AS ENUM ('queued','running','succeeded','failed');
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """)
        
        # Create tables
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS public.source_messages (
                id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
                platform text NOT NULL DEFAULT 'discord',
                guild_id text,
                channel_id text,
                message_id text UNIQUE,
                author_id text,
                posted_at timestamptz,
                raw_json jsonb,
                created_at timestamptz NOT NULL DEFAULT now()
            );
        """)
        
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS public.documents (
                id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
                source_message_id uuid REFERENCES public.source_messages(id) ON DELETE SET NULL,
                url text,
                storage_bucket text,
                storage_path text,
                sha256_hex text UNIQUE,
                title text,
                author text,
                description text,
                content_kind content_kind NOT NULL,
                bytes bigint,
                lang text,
                meta jsonb NOT NULL DEFAULT '{}'::jsonb,
                created_at timestamptz NOT NULL DEFAULT now()
            );
            
            CREATE INDEX IF NOT EXISTS documents_url_idx ON public.documents USING btree(url);
            CREATE INDEX IF NOT EXISTS documents_meta_gin ON public.documents USING gin(meta jsonb_path_ops);
        """)
        
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS public.document_versions (
                id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
                document_id uuid NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
                version integer NOT NULL,
                normalized_format text NOT NULL DEFAULT 'markdown',
                text_len integer NOT NULL,
                text_md text,
                text_plain text,
                tsv tsvector,
                created_at timestamptz NOT NULL DEFAULT now(),
                UNIQUE(document_id, version)
            );
        """)
        
        # Create trigger function for tsvector updates
        await self._conn.execute("""
            CREATE OR REPLACE FUNCTION public.document_versions_tsv_update() RETURNS trigger AS $$
            BEGIN
                NEW.tsv :=
                    setweight(to_tsvector('simple', coalesce(NEW.text_plain,'')), 'A') ||
                    setweight(to_tsvector('simple', coalesce(NEW.text_md,'')), 'B');
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)
        
        await self._conn.execute("""
            DROP TRIGGER IF EXISTS document_versions_tsv_trigger ON public.document_versions;
            CREATE TRIGGER document_versions_tsv_trigger
            BEFORE INSERT OR UPDATE OF text_md, text_plain
            ON public.document_versions
            FOR EACH ROW EXECUTE FUNCTION public.document_versions_tsv_update();
        """)
        
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS document_versions_tsv_idx 
            ON public.document_versions USING gin(tsv);
        """)
        
        # Document chunks with pgvector
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS public.document_chunks (
                id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
                document_version_id uuid NOT NULL REFERENCES public.document_versions(id) ON DELETE CASCADE,
                chunk_index integer NOT NULL,
                char_start integer NOT NULL,
                char_end integer NOT NULL,
                token_count integer,
                text text NOT NULL,
                tsv tsvector,
                embedding vector(768),
                meta jsonb NOT NULL DEFAULT '{}'::jsonb,
                UNIQUE(document_version_id, chunk_index)
            );
        """)
        
        # Trigger for chunks tsvector
        await self._conn.execute("""
            DROP TRIGGER IF EXISTS document_chunks_tsv_trigger ON public.document_chunks;
            CREATE TRIGGER document_chunks_tsv_trigger
            BEFORE INSERT OR UPDATE OF text
            ON public.document_chunks
            FOR EACH ROW EXECUTE FUNCTION public.document_versions_tsv_update();
        """)
        
        # Vector index - use IVFFlat for now
        try:
            await self._conn.execute("""
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_ivfflat
                ON public.document_chunks
                USING ivfflat (embedding vector_l2_ops)
                WITH (lists = 100);
            """)
        except Exception as e:
            logging.warning(f"Could not create IVFFlat index: {e}")
            
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_tsv_idx 
            ON public.document_chunks USING gin(tsv);
        """)
        
        # Collections
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS public.collections (
                id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
                slug text UNIQUE NOT NULL,
                title text,
                meta jsonb NOT NULL DEFAULT '{}'::jsonb,
                created_at timestamptz NOT NULL DEFAULT now()
            );
        """)
        
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS public.collection_membership (
                collection_id uuid REFERENCES public.collections(id) ON DELETE CASCADE,
                document_id uuid REFERENCES public.documents(id) ON DELETE CASCADE,
                PRIMARY KEY (collection_id, document_id)
            );
        """)
        
        # Ingestion jobs
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS public.ingest_jobs (
                id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
                source_message_id uuid REFERENCES public.source_messages(id) ON DELETE SET NULL,
                document_id uuid REFERENCES public.documents(id) ON DELETE SET NULL,
                url text,
                storage_path text,
                status ingest_status NOT NULL DEFAULT 'queued',
                error text,
                timings_ms jsonb,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now()
            );
        """)
        
        logging.info("Postgres schema initialized")
        
    async def store_source_message(
        self,
        guild_id: Optional[str],
        channel_id: str,
        message_id: str,
        author_id: str,
        posted_at: datetime,
        raw_json: dict[str, Any]
    ) -> str:
        """Store Discord source message provenance."""
        msg_id = str(uuid.uuid4())
        
        if self.backend == StorageBackend.SQLITE:
            await self._conn.execute(
                """INSERT INTO source_messages 
                   (id, guild_id, channel_id, message_id, author_id, posted_at, raw_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(message_id) DO UPDATE SET raw_json = excluded.raw_json""",
                (msg_id, guild_id, channel_id, message_id, author_id, 
                 posted_at.isoformat(), json.dumps(raw_json))
            )
            await self._conn.commit()
        elif self.backend == StorageBackend.POSTGRES:
            result = await self._conn.fetchrow(
                """INSERT INTO public.source_messages 
                   (guild_id, channel_id, message_id, author_id, posted_at, raw_json)
                   VALUES ($1, $2, $3, $4, $5, $6)
                   ON CONFLICT (message_id) DO UPDATE SET raw_json = EXCLUDED.raw_json
                   RETURNING id""",
                guild_id, channel_id, message_id, author_id, posted_at, raw_json
            )
            msg_id = str(result['id'])
            
        return msg_id
        
    async def store_document(
        self,
        source_message_id: str,
        url: Optional[str],
        content: bytes,
        title: Optional[str],
        content_kind: ContentKind,
        text_md: Optional[str],
        text_plain: Optional[str],
        chunks: list[DocumentChunk],
        meta: dict[str, Any]
    ) -> str:
        """Store a document with its chunks and embeddings."""
        # Calculate content hash
        sha256_hex = hashlib.sha256(content).hexdigest()
        
        doc_id = str(uuid.uuid4())
        version_id = str(uuid.uuid4())
        
        if self.backend == StorageBackend.SQLITE:
            # Check if document already exists
            cursor = await self._conn.execute(
                "SELECT id FROM documents WHERE sha256_hex = ?",
                (sha256_hex,)
            )
            existing = await cursor.fetchone()
            
            if existing:
                logging.info(f"Document with hash {sha256_hex} already exists")
                return existing[0]
                
            # Insert document
            await self._conn.execute(
                """INSERT INTO documents 
                   (id, source_message_id, url, sha256_hex, title, content_kind, bytes, meta)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, source_message_id, url, sha256_hex, title, 
                 content_kind.value, len(content), json.dumps(meta))
            )
            
            # Insert version
            text_len = len(text_plain or text_md or "")
            await self._conn.execute(
                """INSERT INTO document_versions 
                   (id, document_id, version, text_len, text_md, text_plain)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (version_id, doc_id, 1, text_len, text_md, text_plain)
            )
            
            # Insert chunks
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                embedding_json = json.dumps(chunk.embedding) if chunk.embedding else None
                await self._conn.execute(
                    """INSERT INTO document_chunks 
                       (id, document_version_id, chunk_index, char_start, char_end, 
                        token_count, text, embedding_json, meta)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (chunk_id, version_id, chunk.chunk_index, chunk.char_start, 
                     chunk.char_end, chunk.token_count, chunk.text, 
                     embedding_json, json.dumps(chunk.meta or {}))
                )
                
            await self._conn.commit()
            
            # Update FAISS index if embeddings present
            if chunks and chunks[0].embedding and faiss and np:
                await self._update_faiss_index()
                
        elif self.backend == StorageBackend.POSTGRES:
            # Check if document already exists
            existing = await self._conn.fetchrow(
                "SELECT id FROM public.documents WHERE sha256_hex = $1",
                sha256_hex
            )
            
            if existing:
                logging.info(f"Document with hash {sha256_hex} already exists")
                return str(existing['id'])
                
            # Insert document
            result = await self._conn.fetchrow(
                """INSERT INTO public.documents 
                   (source_message_id, url, sha256_hex, title, content_kind, bytes, meta)
                   VALUES ($1, $2, $3, $4, $5::content_kind, $6, $7)
                   RETURNING id""",
                uuid.UUID(source_message_id), url, sha256_hex, title, 
                content_kind.value, len(content), meta
            )
            doc_id = str(result['id'])
            
            # Insert version
            text_len = len(text_plain or text_md or "")
            result = await self._conn.fetchrow(
                """INSERT INTO public.document_versions 
                   (document_id, version, text_len, text_md, text_plain)
                   VALUES ($1, $2, $3, $4, $5)
                   RETURNING id""",
                uuid.UUID(doc_id), 1, text_len, text_md, text_plain
            )
            version_id = str(result['id'])
            
            # Insert chunks
            for chunk in chunks:
                embedding_vec = chunk.embedding if chunk.embedding else None
                await self._conn.execute(
                    """INSERT INTO public.document_chunks 
                       (document_version_id, chunk_index, char_start, char_end, 
                        token_count, text, embedding, meta)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
                    uuid.UUID(version_id), chunk.chunk_index, chunk.char_start, 
                    chunk.char_end, chunk.token_count, chunk.text, 
                    embedding_vec, chunk.meta or {}
                )
                
        logging.info(f"Stored document {doc_id} with {len(chunks)} chunks")
        return doc_id
        
    async def _update_faiss_index(self) -> None:
        """Update FAISS index from SQLite embeddings."""
        if not faiss or not np:
            return
            
        cursor = await self._conn.execute(
            """SELECT id, embedding_json FROM document_chunks 
               WHERE embedding_json IS NOT NULL"""
        )
        rows = await cursor.fetchall()
        
        if not rows:
            return
            
        embeddings = []
        chunk_ids = []
        
        for row in rows:
            chunk_id, embedding_json = row
            embedding = json.loads(embedding_json)
            embeddings.append(embedding)
            chunk_ids.append(chunk_id)
            
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            dim = embeddings_array.shape[1]
            
            if self._faiss_index is None:
                self._faiss_index = faiss.IndexFlatL2(dim)
                
            self._faiss_index.add(embeddings_array)
            
            # Update chunk ID mapping
            start_idx = self._faiss_index.ntotal - len(chunk_ids)
            for i, chunk_id in enumerate(chunk_ids):
                self._chunk_id_map[start_idx + i] = chunk_id
                
            logging.info(f"Updated FAISS index with {len(embeddings)} embeddings")
            
    async def search_hybrid(
        self,
        query_text: str,
        query_embedding: Optional[list[float]],
        k: int = 10,
        collection_slug: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Perform hybrid search (vector + lexical)."""
        results = []
        
        if self.backend == StorageBackend.SQLITE:
            # Lexical search using FTS5
            cursor = await self._conn.execute(
                """SELECT dc.id, dc.text, dc.chunk_index, d.id as doc_id, d.title, d.url,
                          rank as score
                   FROM document_chunks_fts fts
                   JOIN document_chunks dc ON fts.chunk_id = dc.id
                   JOIN document_versions dv ON dc.document_version_id = dv.id
                   JOIN documents d ON dv.document_id = d.id
                   WHERE document_chunks_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query_text, k * 4)
            )
            lexical_results = await cursor.fetchall()
            
            # Vector search using FAISS
            vector_results = []
            if query_embedding and self._faiss_index and self._faiss_index.ntotal > 0:
                query_vec = np.array([query_embedding], dtype=np.float32)
                distances, indices = self._faiss_index.search(query_vec, min(k * 4, self._faiss_index.ntotal))
                
                for dist, idx in zip(distances[0], indices[0]):
                    if idx in self._chunk_id_map:
                        chunk_id = self._chunk_id_map[idx]
                        cursor = await self._conn.execute(
                            """SELECT dc.id, dc.text, dc.chunk_index, d.id as doc_id, d.title, d.url
                               FROM document_chunks dc
                               JOIN document_versions dv ON dc.document_version_id = dv.id
                               JOIN documents d ON dv.document_id = d.id
                               WHERE dc.id = ?""",
                            (chunk_id,)
                        )
                        row = await cursor.fetchone()
                        if row:
                            vector_results.append({
                                'chunk_id': row[0],
                                'text': row[1],
                                'chunk_index': row[2],
                                'doc_id': row[3],
                                'title': row[4],
                                'url': row[5],
                                'distance': float(dist),
                                'vscore': 1.0 / (1.0 + float(dist))
                            })
                            
            # Combine results with simple fusion
            combined = {}
            for row in lexical_results:
                chunk_id = row[0]
                combined[chunk_id] = {
                    'chunk_id': chunk_id,
                    'text': row[1],
                    'chunk_index': row[2],
                    'doc_id': row[3],
                    'title': row[4],
                    'url': row[5],
                    'lscore': abs(float(row[6])),
                    'vscore': 0.0
                }
                
            for vr in vector_results:
                chunk_id = vr['chunk_id']
                if chunk_id in combined:
                    combined[chunk_id]['vscore'] = vr['vscore']
                else:
                    combined[chunk_id] = vr
                    combined[chunk_id]['lscore'] = 0.0
                    
            # Calculate final scores
            for chunk_id in combined:
                combined[chunk_id]['score'] = (
                    combined[chunk_id].get('vscore', 0.0) * 0.6 + 
                    combined[chunk_id].get('lscore', 0.0) * 0.4
                )
                
            results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)[:k]
            
        elif self.backend == StorageBackend.POSTGRES:
            # Use the hybrid query from the problem statement
            query = """
                WITH
                vec AS (
                    SELECT
                        dc.id AS chunk_id,
                        dc.document_version_id,
                        dv.document_id,
                        (dc.embedding <-> $2) AS dist,
                        1.0 / (1.0 + (dc.embedding <-> $2)) AS vscore
                    FROM public.document_chunks dc
                    JOIN public.document_versions dv ON dv.id = dc.document_version_id
                    WHERE $2 IS NOT NULL
                    ORDER BY dc.embedding <-> $2
                    LIMIT $3 * 4
                ),
                lex AS (
                    SELECT
                        dc.id AS chunk_id,
                        dc.document_version_id,
                        dv.document_id,
                        ts_rank(dc.tsv, plainto_tsquery('simple', unaccent($1))) AS lscore
                    FROM public.document_chunks dc
                    JOIN public.document_versions dv ON dv.id = dc.document_version_id
                    WHERE dc.tsv @@ plainto_tsquery('simple', unaccent($1))
                    ORDER BY lscore DESC
                    LIMIT $3 * 4
                ),
                fused AS (
                    SELECT chunk_id, document_version_id, document_id,
                           COALESCE(MAX(vscore), 0) * 0.6 + COALESCE(MAX(lscore), 0) * 0.4 AS score
                    FROM (
                        SELECT chunk_id, document_version_id, document_id, vscore, 0.0 as lscore FROM vec
                        UNION ALL
                        SELECT chunk_id, document_version_id, document_id, 0.0 as vscore, lscore FROM lex
                    ) u
                    GROUP BY chunk_id, document_version_id, document_id
                )
                SELECT f.chunk_id, f.score,
                       d.id AS document_id, d.title, d.url,
                       dv.version, dc.chunk_index, dc.text
                FROM fused f
                JOIN public.document_chunks dc ON dc.id = f.chunk_id
                JOIN public.document_versions dv ON dv.id = f.document_version_id
                JOIN public.documents d ON d.id = f.document_id
                ORDER BY f.score DESC
                LIMIT $3
            """
            
            rows = await self._conn.fetch(query, query_text, query_embedding, k)
            
            for row in rows:
                results.append({
                    'chunk_id': str(row['chunk_id']),
                    'score': float(row['score']),
                    'doc_id': str(row['document_id']),
                    'title': row['title'],
                    'url': row['url'],
                    'chunk_index': row['chunk_index'],
                    'text': row['text']
                })
                
        return results
