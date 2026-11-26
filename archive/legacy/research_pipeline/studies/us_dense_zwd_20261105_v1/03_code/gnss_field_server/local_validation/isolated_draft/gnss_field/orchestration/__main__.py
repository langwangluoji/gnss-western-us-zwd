"""CPU afterany summary entry point."""
import argparse
import json

from .scheduler import Coordinator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('status',))
    parser.add_argument('--campaign-dir', required=True)
    args = parser.parse_args()
    result = Coordinator.from_directory(args.campaign_dir).status(write=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
