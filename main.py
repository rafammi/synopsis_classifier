import argparse


def main():
    parser = argparse.ArgumentParser(description="Movie genre classifier CLI")
    sub = parser.add_subparsers(dest="command", required = True)
    sub.add_parser("train", help = "Train the LSTM model")

    predict_p = sub.add_parser("predict", help="Predict genres for specified synopsis")
    predict_p.add_argument("--text", type = str, help = "Single movie synopsis")
    predict_p.add_argument("--batch", nargs = "+", help = "Multiple synopses")

    parser_p = sub.add_parser("parse", help="Fetch and save dataset from TMDB")
    parser_p.add_argument("--API_KEY", required=True, help = "TMDB API key")
    parser_p.add_argument("--start_year", type=int, required=True, help = "Start year for parsing")
    parser_p.add_argument("--end_year", type=int, required=True, help = "End year for parsing")
    parser_p.add_argument("-m", "--max_limit", type=int, default=50000, help = "Max number of movies to fetch")

    args = parser.parse_args()

    if args.command == "train":
        from scripts.train import train_pipeline
        train_pipeline()
    elif args.command == "predict":
        from scripts.predict import predict_pipeline
        overview = args.batch if args.batch else args.text 
        predict_pipeline(overview)
    elif args.command == "parse":
        from parser.parser import run_parser
        run_parser(args.API_KEY, args.start_year, args.end_year, args.max_limit)

if __name__ == "__main__":
    main()
