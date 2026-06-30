import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes.prediction import router as prediction_router
from app.routes.prediction import train_router, batch_router, rank_router
from app.services.train_model import MODEL_PATH, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

ALLOWED_ORIGINS: list[str] = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5000,http://localhost:7000",
).split(",")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        logger.info(
            "No trained model found at '%s'. Running initial training...", MODEL_PATH
        )
        try:
            metrics = train()
            logger.info(
                "Auto-training complete — accuracy=%.4f, auc_roc=%.4f  "
                "(split: %s, cv_avg_accuracy=%.4f)",
                metrics["test_metrics"]["accuracy"],
                metrics["test_metrics"]["auc_roc"],
                metrics["split_method"],
                metrics["cv_results"].get("avg_accuracy") or 0.0,
            )
        except Exception as exc:
            logger.error(
                "Auto-training failed: %s. Start the service and call POST /train manually.",
                exc,
            )
    else:
        logger.info("Existing model found at '%s'. Skipping auto-train.", MODEL_PATH)

    yield  

    logger.info("ML service shutting down.")

def create_app() -> FastAPI:
    app = FastAPI(
        title="Trinova ML Service",
        description=(
            "AI microservice for the Trinova ERP platform. "
            "Provides supplier delay risk prediction and on-demand model retraining "
            "using an XGBoost classifier trained on historical procurement data."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(prediction_router)
    app.include_router(batch_router)
    app.include_router(train_router)
    app.include_router(rank_router)

    @app.get("/health", tags=["Health"], summary="Service health check")
    def health_check():
        model_ready = MODEL_PATH.exists()
        return JSONResponse(
            content={
                "status": "ok",
                "model_ready": model_ready,
                "model_path": str(MODEL_PATH),
            }
        )

    @app.get("/", tags=["Health"], include_in_schema=False)
    def root():
        return {"service": "trinova-ml-service", "version": "1.0.0", "docs": "/docs"}

    return app


app = create_app()
