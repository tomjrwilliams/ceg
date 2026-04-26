
import argparse
import ceg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("app", type=str)
    args, _ = parser.parse_known_args()
    app = ceg.app.dev.APPS[args.app].run()