from __future__ import annotations

import argparse
import json

from .service_factory import get_rag_service


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--title", default=None)
    search.add_argument("--top-k", type=int, default=5)

    ask = sub.add_parser("ask")
    ask.add_argument("query")
    ask.add_argument("--title", default=None)

    args = parser.parse_args()
    rag = get_rag_service()

    if args.command == "search":
        result = rag.retrieve(args.query, top_k=args.top_k, title=args.title)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        result = rag.answer(args.query, title=args.title)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
