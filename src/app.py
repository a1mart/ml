from fastapi import FastAPI
from src.routes import datasets, models, inferences

app = FastAPI(title="ML API",
    description="""
    Machine Learning API

    - Upload datasets
    - Register and train ML models
    - Evaluate and rank models
    - Make predictions
    - List models and view reports
    """,
        version="1.0.0",
        contact={
            "name": "AM",
            "email": "aidan3martin@gmail.com",
            "url": "https://github.com/a1mart/ml"
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        },
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

app.include_router(datasets.router)
app.include_router(models.router)
app.include_router(inferences.router)