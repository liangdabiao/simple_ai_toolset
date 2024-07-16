# optimized
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from aiohttp import ClientSession
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

import openai
from starlette.applications import Starlette
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
from starlette.types import ASGIApp

from .routers import tools
from .helpers import record_log, LogLevel,get_calling_function_name ,get_calling_module_name



# Assumed imports from your other modules
from .routers import (
    tools
)
from .mongo import establish_connection, mongo_db_instance
from .logger_setup import initialize_logger
from .config_loader import config



API_KEY_NAME = config["API_KEY_HEADER_NAME"]
API_KEY = config["API_KEY_PASSPHRASE"]
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

WHITELISTED_PATHS = ["/customdocs", "/openapi.json", "/"]



app = FastAPI(
    title="API System",
    description="Private APIs For YourName",
    version="1.0.0",
    servers=[
        {"url": "https://api.url/", "description": "Production server"},
        {"url": "http://127.0.0.1:8080", "description": "Development server"}
    ],
    docs_url="/customdocs"
)

#app = FastAPI(docs_url="/customdocs")


limiter = Limiter(key_func=get_remote_address, default_limits=["1000/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Logger setup at top level
logger = initialize_logger()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Middleware for Logging with enhanced context
class LogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(f"Unhandled error in request {request.url.path}: {exc}", exc_info=True)
            raise exc
        logger.info(f"Request {request.method} {request.url.path}, Response: {response.status_code}")
        return response

app.add_middleware(LogMiddleware)



class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path not in WHITELISTED_PATHS:
            api_key = request.headers.get(API_KEY_NAME)
            if api_key != API_KEY:
                return JSONResponse(
                    status_code=401, content={"detail": "Invalid API key"}
                )
        return await call_next(request)
    
app.add_middleware(APIKeyMiddleware)

original_openapi = app.openapi


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = original_openapi()
    openapi_schema["components"]["securitySchemes"] = {
        API_KEY_NAME: {"type": "apiKey", "in": "header", "name": API_KEY_NAME}
    }
    openapi_schema["security"] = [{API_KEY_NAME: []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi




@app.on_event("startup")
async def startup():
    try:
        print("establishing connection to MongoDB")
        #await establish_connection()
    except Exception as ex:
        record_log(ex,get_calling_module_name(),get_calling_function_name(), LogLevel.ERROR)
        #logger.error(f"Error occurred on startup_mongoObj.connect(): {e}", exc_info=True)
        


@app.on_event("shutdown")
async def shutdown():
    try:
        print("close connection to MongoDB")
        #await mongo_db_instance.close()
    except Exception as e:
        logger.warning(f"error occurred on shutdown_MongoDB.close(): {e}", exc_info=True)



@app.get("/health")
async def read_health():
    try:
        # Assuming mongoObj has a method to check connection health
        #if not await mongo_db_instance.is_connected():
            #raise Exception("MongoDB is disconnected")
        
        # Additional checks can go here
        return {"status": "healthy"}

    except Exception as ex:
        # Log the specific error for internal reference
        record_log(ex,get_calling_module_name(),get_calling_function_name(), LogLevel.ERROR)
        # Return an appropriate error status code
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "details": str(ex)}
        )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    try:

        logger.error(
            f"An HTTP error occurred: {exc.detail}",
            extra={"request_path": request.url.path}
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"message": exc.detail},
        )
    except:
        return
        #nothing





@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    try:
        logger.error(
            f"An unexpected error occurred while processing path {request.url.path}: {exc}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "An unexpected error occurred",
                "result": None,
            },
        )
    except:
        return
        #nothing




app.include_router(tools.router, tags=["tools"])
