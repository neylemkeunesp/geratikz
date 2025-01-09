from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import openai
import os
import tempfile
import base64
import logging
import requests
import json
from pathlib import Path
from latex import build_pdf
from pdf2image import convert_from_path
from dotenv import load_dotenv

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,  # Set default level to INFO to reduce noise
    format='%(asctime)s - %(levelname)s - %(message)s',  # Simplified format
    handlers=[
        logging.FileHandler('geratikz.log'),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

# Set specific logger for our application
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Keep debug level for our app

# Reduce noise from other loggers
logging.getLogger("python_multipart").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("fastapi").setLevel(logging.INFO)

logger.info("Logging initialized - Check geratikz.log for detailed logs")

# Load environment variables from .env file
load_dotenv()

# Debug log API key loading
# api_key = os.getenv('OPENAI_API_KEY')
#get api from .env file
api_key = os.getenv('OPENROUTER_API_KEY')
if api_key:
    logger.info(f"Loaded API key: {api_key[:10]}...")
else:
    logger.error("No API key found in environment variables")

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def generate_tikz(description: str, model: str = "anthropic/claude-3-haiku") -> str:
    """Generate TikZ code from text description using OpenRouter.
    
    Args:
        description: The text description of the figure to generate
        model: The model to use (default: anthropic/claude-3-haiku)
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "HTTP-Referer": "http://localhost:3333",  # Optional but good practice
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": """You are a LaTeX/TikZ expert. Generate ONLY the TikZ code that matches the given description.
Rules:
1. Do not include any explanatory text or comments
2. Do not include LaTeX preamble or document environment
3. Only include the TikZ commands needed to draw the figure
4. If color is specified, use the appropriate TikZ color command
5. Use exact measurements as specified in the description
Example for 'Draw a red circle with radius 2cm centered at origin':
\\draw[red] (0,0) circle (2cm);"""},
            {"role": "user", "content": description}
        ]
    }

    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        logger.info(f"Making request to OpenRouter API: {url}")
        logger.info("Headers: %s", {k: v[:10] + '...' if k == 'Authorization' else v for k, v in headers.items()})
        logger.info("Payload: %s", json.dumps(payload, indent=2))
        
        response = requests.post(
            url,
            headers=headers,
            json=payload
        )
        
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        logger.info(f"Response text: {response.text}")
        
        if response.status_code != 200:
            raise Exception(f"OpenRouter API error: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        raise
    
    response_json = response.json()
    logger.info("OpenRouter response: %s", response_json)  # Debug print
    
    if "choices" in response_json:
        tikz_code = response_json["choices"][0]["message"]["content"]
        # Ensure the TikZ code is wrapped in a tikzpicture environment
        if not "\\begin{tikzpicture}" in tikz_code:
            tikz_code = "\\begin{tikzpicture}\n" + tikz_code + "\n\\end{tikzpicture}"
        return tikz_code
    elif "error" in response_json:
        raise Exception(f"OpenRouter API error: {response_json['error']}")
    else:
        raise Exception(f"Unexpected response format: {response_json}")

def compile_tikz(tikz_code: str) -> Path:
    """Compile TikZ code to PDF and convert to PNG."""
    latex_template = r"""
    \documentclass[tikz,border=10pt]{standalone}
    \usepackage{tikz}
    \usetikzlibrary{calc}
    \begin{document}
    %s
    \end{document}
    """ % tikz_code

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create a temporary file with the LaTeX content
        tex_path = tmp_path / "figure.tex"
        with open(tex_path, "w") as f:
            f.write(latex_template)
        
        # Run pdflatex with shell-escape enabled and capture output
        import subprocess
        try:
            result = subprocess.run(
                ['pdflatex', '-shell-escape', str(tex_path)],
                cwd=tmp_path,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("LaTeX compilation output: %s", result.stdout)
            logger.info("LaTeX compilation errors: %s", result.stderr)
        except subprocess.CalledProcessError as e:
            logger.error("LaTeX compilation failed: %s %s", e.stdout, e.stderr)
            raise Exception(f"LaTeX compilation failed: {e.stdout}\n{e.stderr}")
        
        # The PDF is already generated at the expected location
        pdf_path = tmp_path / "figure.pdf"
        
        # Convert PDF to PNG
        images = convert_from_path(tmp_path / "figure.pdf")
        png_path = Path("static/figures/latest.png")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(png_path, "PNG")
        return png_path

def get_available_models() -> list[dict]:
    """Fetch available models from OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "HTTP-Referer": "http://localhost:3333",
        "Content-Type": "application/json"
    }
    
    try:
        logger.info("Fetching models from OpenRouter API...")
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        
        models = response.json()
        logger.info(f"Retrieved {len(models)} models from OpenRouter")
        
        # Filter for models that support chat completion
        chat_models = []
        for model in models:
            types = model.get("types", [])
            context_length = model.get("context_length", 0)
            
            # Only include models that:
            # 1. Support chat completion
            # 2. Have a reasonable context length (>1000 tokens)
            # 3. Have valid pricing information
            if (any(t == "chat" for t in types) and 
                context_length >= 1000 and 
                model.get("pricing", {}).get("prompt") is not None):
                chat_models.append(model)
        
        logger.info(f"Filtered to {len(chat_models)} valid chat models")
        
        # Sort by pricing (cost per 1k tokens)
        chat_models.sort(key=lambda x: x.get("pricing", {}).get("prompt", 0))
        
        # Log the available models for debugging
        for model in chat_models:
            logger.info(f"Available model: {model.get('id')} - "
                       f"Context: {model.get('context_length')} tokens, "
                       f"Price: {model.get('pricing', {}).get('prompt', 0):.5f} per token")
        
        return chat_models
        
    except requests.exceptions.Timeout:
        logger.error("Timeout while fetching models from OpenRouter")
        return get_fallback_models()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed while fetching models: {str(e)}")
        return get_fallback_models()
    except Exception as e:
        logger.error(f"Unexpected error while fetching models: {str(e)}")
        return get_fallback_models()

def get_fallback_models() -> list[dict]:
    """Return a list of fallback models when OpenRouter API is unavailable."""
    return [
        {
            "id": "anthropic/claude-3-opus-20240229",
            "name": "Claude 3 Opus",
            "pricing": {"prompt": 0.00150}
        },
        {
            "id": "anthropic/claude-3-sonnet-20240229",
            "name": "Claude 3 Sonnet",
            "pricing": {"prompt": 0.00075}
        },
        {
            "id": "anthropic/claude-3-haiku-20240307",
            "name": "Claude 3 Haiku",
            "pricing": {"prompt": 0.00025}
        },
        {
            "id": "anthropic/claude-2.1",
            "name": "Claude 2.1",
            "pricing": {"prompt": 0.00080}
        },
        {
            "id": "anthropic/claude-2.0",
            "name": "Claude 2.0",
            "pricing": {"prompt": 0.00080}
        },
        {
            "id": "anthropic/claude-instant-1.2",
            "name": "Claude Instant 1.2",
            "pricing": {"prompt": 0.00020}
        },
        {
            "id": "google/gemini-pro",
            "name": "Gemini Pro",
            "pricing": {"prompt": 0.00010}
        },
        {
            "id": "google/gemini-pro-vision",
            "name": "Gemini Pro Vision",
            "pricing": {"prompt": 0.00010}
        },
        {
            "id": "meta-llama/llama-2-13b-chat",
            "name": "Llama 2 13B",
            "pricing": {"prompt": 0.00010}
        },
        {
            "id": "meta-llama/llama-2-70b-chat",
            "name": "Llama 2 70B",
            "pricing": {"prompt": 0.00015}
        },
        {
            "id": "mistral/mistral-tiny",
            "name": "Mistral Tiny",
            "pricing": {"prompt": 0.00010}
        },
        {
            "id": "mistral/mistral-small",
            "name": "Mistral Small",
            "pricing": {"prompt": 0.00020}
        },
        {
            "id": "mistral/mistral-medium",
            "name": "Mistral Medium",
            "pricing": {"prompt": 0.00040}
        },
        {
            "id": "perplexity/pplx-7b-chat",
            "name": "PPLX 7B",
            "pricing": {"prompt": 0.00010}
        },
        {
            "id": "perplexity/pplx-70b-chat",
            "name": "PPLX 70B",
            "pricing": {"prompt": 0.00020}
        },
        {
            "id": "perplexity/pplx-online",
            "name": "PPLX Online",
            "pricing": {"prompt": 0.00020}
        },
        {
            "id": "openai/gpt-4",
            "name": "GPT-4",
            "pricing": {"prompt": 0.00300}
        },
        {
            "id": "openai/gpt-4-turbo",
            "name": "GPT-4 Turbo",
            "pricing": {"prompt": 0.00100}
        },
        {
            "id": "openai/gpt-4-vision",
            "name": "GPT-4 Vision",
            "pricing": {"prompt": 0.00300}
        },
        {
            "id": "openai/gpt-3.5-turbo",
            "name": "GPT-3.5 Turbo",
            "pricing": {"prompt": 0.00015}
        }
    ]

def validate_figure(description: str, image_path: Path, model: str = "anthropic/claude-3-haiku") -> tuple[bool, str]:
    """Use OpenRouter Vision to analyze and validate the figure.
    
    Args:
        description: The text description to validate against
        image_path: Path to the generated figure image
        model: The model to use for validation (default: anthropic/claude-3-haiku)
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "HTTP-Referer": "http://localhost:3333",  # Optional but good practice
        "Content-Type": "application/json"
    }

    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode()

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Does this TikZ figure accurately represent the following description? Description: {description}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ]
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"OpenRouter API error: {response.text}")
    
    response_json = response.json()
    logger.info("OpenRouter response: %s", response_json)  # Debug print
    if "choices" in response_json:
        response_text = response_json["choices"][0]["message"]["content"]
        logger.info("Image analysis: %s", response_text)  # Debug print
        return "yes" in response_text.lower(), response_text
    elif "error" in response_json:
        raise Exception(f"OpenRouter API error: {response_json['error']}")
    else:
        raise Exception(f"Unexpected response format: {response_json}")

@app.get("/")
async def root():
    return {"message": "Welcome to GeraTikZ"}

@app.get("/ui", response_class=HTMLResponse)
async def home(request: Request):
    logger.info("Handling UI route")  # Debug print
    try:
        models = get_available_models()
        default_model = "anthropic/claude-3-haiku"
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": None,
            "model": default_model,
            "available_models": models
        })
    except Exception as e:
        logger.error(f"Template error: {str(e)}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Template error: {str(e)}",
            "model": "anthropic/claude-3-haiku"  # Set default model
        })

@app.post("/generate")
async def generate(request: Request, description: str = Form(...), model: str = Form(...)):
    try:
        # Check if OpenRouter API key is valid
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key or not api_key.startswith("sk-or-"):
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "error": "Invalid OpenRouter API key. Please check your configuration. Key should start with 'sk-or-'",
                    "description": description,
                    "model": model,
                    "available_models": get_available_models()
                }
            )
            
        tikz_code = generate_tikz(description, model)
        logger.info("Generated TikZ code: %s", tikz_code)  # Debug print
        image_path = compile_tikz(tikz_code)
        is_valid, analysis = validate_figure(description, image_path, model)
        
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "tikz_code": tikz_code,
                "image_url": "/static/figures/latest.png",
                "description": description,
                "is_valid": is_valid,
                "analysis": analysis,
                "error": None,
                "model": model,
                "available_models": get_available_models()
            }
        )
    except requests.exceptions.HTTPError as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Invalid OpenRouter API key. Please check your configuration.",
                "description": description,
                "model": model,
                "available_models": get_available_models()
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": str(e),
                "description": description,
                "model": model,
                "available_models": get_available_models()
            }
        )

@app.get("/test")
async def test():
    """Test route that attempts to make an API call to OpenRouter"""
    # Get and validate API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        error_msg = "No API key found in environment variables"
        logger.error(error_msg)
        return {"error": error_msg}
    
    if not api_key.startswith('sk-or-'):
        error_msg = f"Invalid API key format. Key should start with 'sk-or-'. Got key starting with: {api_key[:5]}..."
        logger.error(error_msg)
        return {"error": error_msg}
    
    logger.info(f"Using API key starting with: {api_key[:10]}...")
    logger.info(f"API key length: {len(api_key)}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:3333",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "anthropic/claude-3-haiku",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }
    
    try:
        logger.info("Making test request to OpenRouter API...")
        logger.info(f"Headers (sanitized): {json.dumps({k: (v[:10] + '...' if k == 'Authorization' else v) for k, v in headers.items()}, indent=2)}")
        logger.info(f"Payload: {json.dumps(payload, indent=2)}")
        
        # Make request with detailed error handling
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()  # Raise an error for bad status codes
            
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            logger.info(f"Response text: {response.text}")
            
            return {
                "message": "Test completed successfully",
                "status": response.status_code,
                "headers": dict(response.headers),
                "response": response.text
            }
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            return {"error": error_msg}
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
            
        except requests.exceptions.Timeout as e:
            error_msg = "Request timed out"
            logger.error(error_msg)
            return {"error": error_msg}
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting GeraTikZ server on http://localhost:3333")
    uvicorn.run(app, host="127.0.0.1", port=3333, log_level="debug")
