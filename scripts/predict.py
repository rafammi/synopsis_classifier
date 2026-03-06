from src import model
import pandas as pd
import logging

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s- %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def predict_pipeline(overview: str | list):
    logger.info("=" * 60)
    logger.info("Starting predictions!")
    logger.info("=" * 60)

    genre_predictor = model.genrePredictor(num_classes=0, class_names=[])
    genre_predictor.load_model()
    logger.info("Model loaded!")

    if isinstance(overview, str):
        predicted = genre_predictor.predict(overview)
        logger.info(f"Predicted genres:\n{predicted.to_string()}")
        return predicted

    elif isinstance(overview, list):
        results = pd.DataFrame(
            {text: genre_predictor.predict(text) for text in overview}
        ).T.fillna(0)
        results.index.name = "overview"
        logger.info(f"Batch predictions:\n{results.to_string()}")
        return results
