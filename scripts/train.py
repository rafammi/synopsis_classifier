from src import model, preprocessing
import argparse
import logging

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s- %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_pipeline():
    logger.info("=" * 60)
    logger.info("Model training initialized")
    logger.info("=" * 60)


    logger.info("=" * 60)
    logger.info("### DATA ###")
    logger.info("=" * 60)
    logger.info("## DATA LOADING ##")
    lst = preprocessing.get_data()
    logger.info("Data parsed.")
    movies = preprocessing.load_all_data(lst)
    logger.info("Data loaded.")

    logger.info("## LABEL PREPROCESSING ##")
    movies = preprocessing.dedup_data(movies)
    movies = preprocessing.normalize_genres(movies)
    y, classes = preprocessing.create_labels(movies)
    logger.info("Labels created.")

    logger.info("## FEATURE PREPROCESSING ##")
    X = preprocessing.prepare_features(movies)
    logger.info("Features created.")
    
    logger.info("## DATA SPLIT ##")
    X_train, X_test, y_train, y_test = preprocessing.split_data(X,y)
    logger.info("Data split into train/test.")
    logger.info(f"Training shape: {X_train.shape, y_train.shape}")
    logger.info(f"Test shape: {X_test.shape, y_test.shape}")

    logger.info("=" * 60)
    logger.info("### MODEL ###")
    logger.info("=" * 60)

    num_samples = y.shape[0]
    class_weights = {i: num_samples / (y.shape[1] * y[:, i].sum()) for i in range(y.shape[1])}
    genrePredictor = model.genrePredictor(len(classes), classes)
    logger.info("Predictor class created.")

    genrePredictor.initialize(X_train)
    genrePredictor.compile()
    genrePredictor.train(X_train,y_train,  class_weights=class_weights)
    genrePredictor.plot_training()

    y_prob_test = genrePredictor.predict_proba(X_test)
    genrePredictor.tune_thresholds(y_prob_test, y_test)
    logger.info(f"Tuned thresholds: {genrePredictor.thresholds}")

    f1_macro, f1_micro, hamming_acc = genrePredictor.evaluate(X_test, y_test)
    logger.info(f"F1 (MACRO)_: {f1_macro}")
    logger.info(f"F1 (MICRO)_: {f1_micro}")
    logger.info(f"hamming_acc: {hamming_acc}")


    genrePredictor.save_model()