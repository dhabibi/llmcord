import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Literal, Optional

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import yaml

from copilot_auth import find_copilot_token

# Optional ingestion imports
try:
    from ingest_db import IngestDB, StorageBackend, ContentKind, DocumentChunk
    from ingest_extractors import ContentExtractor, create_chunks
    from ingest_embeddings import EmbeddingGenerator, estimate_tokens
    INGEST_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Ingestion features not available: {e}")
    INGEST_AVAILABLE = False
    IngestDB = None
    ContentExtractor = None
    EmbeddingGenerator = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

DEFAULT_API_KEY = "sk-no-key-required"

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0

# Ingestion system (optional)
ingest_db = None
ingest_extractor = None
ingest_embedder = None

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]

    return choices[:25]


@discord_bot.tree.command(name="ingest", description="Ingest links and attachments into knowledge base")
async def ingest_command(interaction: discord.Interaction) -> None:
    """Ingest all links and file attachments from recent messages in this channel."""
    global ingest_db, ingest_extractor, ingest_embedder
    
    if not INGEST_AVAILABLE:
        await interaction.response.send_message(
            "❌ Ingestion feature not available. Install required dependencies: "
            "`pip install aiosqlite beautifulsoup4 html2text faiss-cpu numpy`",
            ephemeral=True
        )
        return
        
    # Check if ingestion is enabled in config
    config = await asyncio.to_thread(get_config)
    ingest_config = config.get("ingest", {})
    
    if not ingest_config.get("enabled", False):
        await interaction.response.send_message(
            "❌ Ingestion is not enabled. Set `ingest.enabled: true` in config.yaml",
            ephemeral=True
        )
        return
        
    # Check permissions
    user_is_admin = interaction.user.id in config["permissions"]["users"]["admin_ids"]
    if not user_is_admin:
        await interaction.response.send_message(
            "❌ You don't have permission to use this command.",
            ephemeral=True
        )
        return
        
    # Initialize ingestion system if needed
    if ingest_db is None:
        try:
            backend_str = ingest_config.get("backend", "sqlite")
            backend = StorageBackend.SQLITE if backend_str == "sqlite" else StorageBackend.POSTGRES
            connection_string = ingest_config.get("connection_string", "ingest.db")
            
            ingest_db = IngestDB(backend, connection_string)
            await ingest_db.connect()
            
            ingest_extractor = ContentExtractor(httpx_client)
            
            # Initialize embedder
            embed_config = ingest_config.get("embedding", {})
            embed_provider = embed_config.get("provider", "openai")
            embed_model = embed_config.get("model", "text-embedding-3-small")
            embed_dim = embed_config.get("dimension", 768)
            
            # Use custom base_url/api_key or fall back to provider config
            embed_base_url = embed_config.get("base_url")
            embed_api_key = embed_config.get("api_key")
            
            if not embed_base_url and embed_provider in config.get("providers", {}):
                provider_config = config["providers"][embed_provider]
                embed_base_url = provider_config.get("base_url")
                embed_api_key = provider_config.get("api_key")
                
            ingest_embedder = EmbeddingGenerator(
                provider=embed_provider,
                model=embed_model,
                base_url=embed_base_url,
                api_key=embed_api_key,
                dimension=embed_dim
            )
            
            logging.info("Ingestion system initialized")
            
        except Exception as e:
            logging.exception("Error initializing ingestion system")
            await interaction.response.send_message(
                f"❌ Failed to initialize ingestion system: {e}",
                ephemeral=True
            )
            return
            
    # Defer response since this might take a while
    await interaction.response.defer(ephemeral=True)
    
    try:
        # Get recent messages from the channel (last 100)
        messages = []
        async for message in interaction.channel.history(limit=100):
            messages.append(message)
            
        # Extract all URLs and attachments
        items_to_ingest = []
        
        for message in messages:
            # Store source message
            source_msg_id = await ingest_db.store_source_message(
                guild_id=str(message.guild.id) if message.guild else None,
                channel_id=str(message.channel.id),
                message_id=str(message.id),
                author_id=str(message.author.id),
                posted_at=message.created_at,
                raw_json={
                    "content": message.content,
                    "author": str(message.author),
                    "channel": str(message.channel)
                }
            )
            
            # Extract URLs from message content
            import re
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            urls = re.findall(url_pattern, message.content)
            
            for url in urls:
                items_to_ingest.append({
                    "type": "url",
                    "source": url,
                    "source_msg_id": source_msg_id,
                    "message": message
                })
                
            # Extract attachments
            for attachment in message.attachments:
                items_to_ingest.append({
                    "type": "attachment",
                    "source": attachment,
                    "source_msg_id": source_msg_id,
                    "message": message
                })
                
        if not items_to_ingest:
            await interaction.followup.send("ℹ️ No links or attachments found in recent messages.")
            return
            
        # Process items
        ingested_count = 0
        failed_count = 0
        skipped_count = 0
        
        status_msg = await interaction.followup.send(
            f"📥 Ingesting {len(items_to_ingest)} items... (0/{len(items_to_ingest)})"
        )
        
        chunk_size = ingest_config.get("chunk_size", 900)
        chunk_overlap = ingest_config.get("chunk_overlap", 120)
        
        for i, item in enumerate(items_to_ingest):
            try:
                # Extract content
                if item["type"] == "url":
                    url = item["source"]
                    logging.info(f"Extracting content from URL: {url}")
                    
                    raw_content, markdown_text, plain_text, title = await ingest_extractor.extract_from_url(url)
                    content_kind = ContentKind.WEBPAGE
                    
                elif item["type"] == "attachment":
                    attachment = item["source"]
                    logging.info(f"Extracting content from attachment: {attachment.filename}")
                    
                    url = attachment.url
                    att_content = await attachment.read()
                    
                    raw_content, markdown_text, plain_text, title = await ingest_extractor.extract_from_attachment(
                        att_content,
                        attachment.filename,
                        attachment.content_type or "application/octet-stream"
                    )
                    
                    # Determine content kind from content type
                    if attachment.content_type:
                        if attachment.content_type.startswith('text/'):
                            content_kind = ContentKind.CODE
                        elif attachment.content_type == 'application/pdf':
                            content_kind = ContentKind.PDF
                        elif attachment.content_type.startswith('image/'):
                            content_kind = ContentKind.IMAGE
                        else:
                            content_kind = ContentKind.OTHER
                    else:
                        content_kind = ContentKind.OTHER
                else:
                    continue
                    
                # Create chunks
                text_to_chunk = plain_text or markdown_text or ""
                if not text_to_chunk.strip():
                    logging.warning(f"No text content extracted from {url}")
                    skipped_count += 1
                    continue
                    
                chunk_tuples = create_chunks(text_to_chunk, chunk_size, chunk_overlap)
                
                if not chunk_tuples:
                    logging.warning(f"No chunks created for {url}")
                    skipped_count += 1
                    continue
                    
                # Generate embeddings for chunks
                chunk_texts = [chunk_text for _, _, chunk_text in chunk_tuples]
                embeddings = await ingest_embedder.generate_embeddings_batch(chunk_texts)
                
                # Create DocumentChunk objects
                chunks = []
                for j, ((start, end, text), embedding) in enumerate(zip(chunk_tuples, embeddings)):
                    token_count = estimate_tokens(text)
                    chunks.append(DocumentChunk(
                        chunk_index=j,
                        char_start=start,
                        char_end=end,
                        text=text,
                        embedding=embedding,
                        token_count=token_count
                    ))
                    
                # Store document
                doc_id = await ingest_db.store_document(
                    source_message_id=item["source_msg_id"],
                    url=url if item["type"] == "url" else None,
                    content=raw_content,
                    title=title,
                    content_kind=content_kind,
                    text_md=markdown_text,
                    text_plain=plain_text,
                    chunks=chunks,
                    meta={
                        "channel_id": str(item["message"].channel.id),
                        "message_id": str(item["message"].id)
                    }
                )
                
                ingested_count += 1
                logging.info(f"Ingested document {doc_id} with {len(chunks)} chunks")
                
            except Exception as e:
                logging.exception(f"Error ingesting item: {e}")
                failed_count += 1
                
            # Update status every 5 items
            if (i + 1) % 5 == 0 or (i + 1) == len(items_to_ingest):
                await status_msg.edit(
                    content=f"📥 Ingesting... ({i + 1}/{len(items_to_ingest)}) | "
                            f"✅ {ingested_count} ingested | ❌ {failed_count} failed | ⏭️ {skipped_count} skipped"
                )
                
        # Final summary
        summary = (
            f"✅ Ingestion complete!\n\n"
            f"**Results:**\n"
            f"• {ingested_count} documents ingested\n"
            f"• {failed_count} failed\n"
            f"• {skipped_count} skipped (no content)\n"
            f"• Total items processed: {len(items_to_ingest)}"
        )
        
        await status_msg.edit(content=summary)
        
    except Exception as e:
        logging.exception("Error during ingestion")
        await interaction.followup.send(f"❌ Error during ingestion: {e}", ephemeral=True)


@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = config["providers"][provider]

    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", DEFAULT_API_KEY)
    
    # Special handling for GitHub Copilot provider
    if provider == "copilot" and api_key == DEFAULT_API_KEY:
        if copilot_token := find_copilot_token():
            api_key = copilot_token
        else:
            logging.warning("GitHub Copilot token not found. Please authenticate with an IDE or set GITHUB_TOKEN environment variable.")
    
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(provider_slash_model.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        if accept_usernames:
            system_prompt += "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."

        messages.append(dict(role="system", content=system_prompt))

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []

    openai_kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                if finish_reason != None:
                    break

                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue

                finish_reason = choice.finish_reason

                prev_content = curr_content or ""
                curr_content = choice.delta.content or ""

                new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time

                    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason != None or msg_split_incoming
                    is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                        else:
                            await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                            await response_msgs[-1].edit(embed=embed)

                        last_task_time = datetime.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

    except Exception:
        logging.exception("Error while generating response")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main() -> None:
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
