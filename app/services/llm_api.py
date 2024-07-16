# optimized
# External imports
import asyncio
import openai
import tiktoken
from datetime import datetime, timedelta, timezone
import logging

# Relative imports
from ..mongo import get_database
from ..words_filter import check_user_input
from ..config_loader import config
from openai import AsyncOpenAI





logger = logging.getLogger("AppLogger")

# Configurations
MAX_RETRIES = config.get("LLM_MAX_RETRIES", 5)
RETRY_DELAY = config.get("LLM_RETRY_DELAY")
API_KEY = config.get("OPENAI_API_KEY")
MODEL = config.get("TXT_SELECTED_MODEL")
MODEL_FUNCTION = config.get("TXT_FUNCTION_SELECTED_MODEL")
API_USED = config.get("TXT_API_USED")
CACHE_ENABLED = config.get("TXT_TOOLS_GENERATION_ENABLE_CACHE")
CACHE_DURATION_DAYS = config.get("TXT_TOOLS_GENERATION_CACHE_DAYS")

openai.api_key = API_KEY
async_openai_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=config.get("OPENAI_API_BASE")
)

import instructor
instructor.apatch(async_openai_client)


def count_tokens(text):
    """Count the number of tokens for a given text."""
    try:
        encoding = tiktoken.encoding_for_model(MODEL)
        return len(encoding.encode(text))
    except Exception:
        logger.exception("llm:count_tokens exception")
        return 0

async def is_text_flagged(input_text):
    """Check if the given text is flagged by the moderation tool."""
    try:
        if check_user_input(input_text):
            return True
        moderation_result = await async_openai_client.moderations.create(input=input_text)
        return moderation_result.results[0].flagged
    except Exception:
        logger.exception("llm:is_text_flagged exception")
        return False

async def get_cached_response(user_prompt):
    """Retrieve a cached response, if it exists, within the cache duration."""
    try:
        db = await get_database("llm_cache_db")
        cache_collection = db["llm_cache"]

        threshold_date = datetime.now(timezone.utc) - timedelta(
            days=CACHE_DURATION_DAYS
        )
        cached_response = await cache_collection.find_one(
            {
                "user_prompt": user_prompt,
                "recorded_date": {"$gte": threshold_date},
            },
            projection={"response": 1, "_id": 0},
        )

        return cached_response.get("response") if cached_response else None

    except Exception:
        logger.exception(
            f"llm:get_cached_response exception for user_prompt={user_prompt}"
        )
        return None

async def cache_response(user_prompt, result, json_retries, attempt, tool_id, model):
    """Cache the response in the database."""
    try:
        db = await get_database("llm_cache_db")
        cache_collection = db["llm_cache"]

        await cache_collection.insert_one(
            {
                "user_prompt": user_prompt,
                "response": result,
                "api": API_USED,
                "model": model,
                "recorded_date": datetime.now(timezone.utc),
                "json_retries": json_retries,
                "api_attempt": attempt,
                "tool_id": tool_id,
            }
        )

    except Exception:
        logger.exception("llm:cache_response exception")

async def generate_response(user_prompt, model, tool_id):
    """Generates a response using OpenAI API and caches it if enabled."""
    response = None
    if CACHE_ENABLED:
        cached_response = await get_cached_response(user_prompt)
        if cached_response:
            return cached_response

    for attempt in range(MAX_RETRIES):
        try:
            completion = await async_openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
            )
            response = completion.choices[0].message.content

        except Exception as e:
            if attempt < MAX_RETRIES - 1:  # don't wait after the last attempt
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                logger.exception(f"Response generation exception after max retries: {e}")
                return None

        await cache_response(user_prompt, response, 0, attempt, tool_id, MODEL)

        return response

async def generate_function_call_response(
    user_prompt,
    function_name,
    function_description,
    function_parameters,
    json_retries,
    tool_id,
):
    """Generate a response using OpenAI API for a function call and cache it."""

    for attempt in range(MAX_RETRIES):
        try:
            completion = await async_openai_client.chat.completions.create(
                model=MODEL_FUNCTION,
                messages=[{"role": "user", "content": user_prompt}],
                functions=[
                    {
                        "name": function_name,
                        "description": function_description,
                        "parameters": function_parameters,
                    }
                ],
                function_call={"name": function_name},
            )

            await cache_response(
                user_prompt, completion, json_retries, attempt, tool_id, MODEL_FUNCTION
            )
            return completion

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                logger.exception(
                    f"generate_function_call_response exception after max retries, error: {e}, function_name: {function_name}, tool_id: {tool_id}"
                )
                return None

async def generate_with_response_model(user_prompt, tool_id,tool_response_model):
    """Generates a response using OpenAI API and caches it if enabled."""
    response = None
    if CACHE_ENABLED:
        cached_response = await get_cached_response(user_prompt)
        if cached_response:
            return cached_response

    for attempt in range(MAX_RETRIES):
        try:
            resut: tool_response_model = await async_openai_client.chat.completions.create(
                model=MODEL_FUNCTION,
                response_model=tool_response_model,
                messages=[{"role": "user", "content": user_prompt}]
            )

            return resut
        except Exception as e:
            if attempt < MAX_RETRIES - 1:  # don't wait after the last attempt
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                logger.exception(f"Response generation exception after max retries: {e}")
                return None

        await cache_response(user_prompt, response, 0, attempt, tool_id, MODEL)

        return response



