from pymongo.errors import ConnectionFailure
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
from .helpers import record_log, LogLevel, get_calling_function_name , get_calling_module_name
from .config_loader import config

MONGO_CONNECTION_STRING = config.get("MONGO_CONNECTION_STRING")


class AsyncMongoDB:
    _instance = None
    _lock = asyncio.Lock()  # To prevent simultaneous reconnections

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AsyncMongoDB, cls).__new__(cls, *args, **kwargs)
            cls._instance.client = None
        return cls._instance

    async def is_connected(self):
        """Check if we have an active MongoDB connection."""
        if self.client:
            try:
                await self.client.admin.command("ismaster")
                return True
            except ConnectionFailure as ex:
                record_log(ex,get_calling_module_name(),get_calling_function_name(), LogLevel.ERROR)
        return False
    
    async def connect(self, retries=5, delay=1):
        async with self._lock:
            if await self.is_connected():
                return

            for attempt in range(1, retries + 1):
                try:
                    connection_string = MONGO_CONNECTION_STRING
                    if not connection_string:
                        raise ValueError("MongoDB connection string not set in environment variables.")
                    self.client = AsyncIOMotorClient(connection_string)
                    if await self.is_connected():
                        return
                except Exception as ex:
                    record_log(ex,get_calling_module_name(),get_calling_function_name(), LogLevel.ERROR)
                    if attempt < retries:
                        await asyncio.sleep(delay)
                        delay *= 2
                    #else:
                        #raise ex  # Re-raise the exception after the last attempt

    async def get_database(self, db_name):
        await self.connect()  # Ensure connection is established
        return self.client[db_name]

    async def close(self):
        if self.client:
            self.client.close()
            self.client = None

mongo_db_instance = AsyncMongoDB()

async def get_database(db_name):
    try:
        return await mongo_db_instance.get_database(db_name)
    except Exception as ex:
        record_log(ex,get_calling_module_name(),get_calling_function_name(), LogLevel.ERROR)
        raise


async def establish_connection():
    try:
        await mongo_db_instance.connect()
    except Exception as ex:
        record_log(ex,get_calling_module_name(),get_calling_function_name(), LogLevel.ERROR)
        raise


