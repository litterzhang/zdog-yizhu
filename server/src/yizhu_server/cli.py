"""Flask served by Waitress on loopback; Nginx terminates public HTTPS."""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from waitress import serve

from yizhu_server.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Yizhu Flask API with zdog-auth")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8091, type=int)
    parser.add_argument(
        "--reload", action="store_true", help="Development reloader (never enables debugger)"
    )
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    app = create_app()
    if args.reload:
        app.run(host=args.host, port=args.port, debug=False, use_reloader=True)
    else:
        serve(
            app,
            host=args.host,
            port=args.port,
            threads=4,
            trusted_proxy="127.0.0.1",
            trusted_proxy_count=1,
            trusted_proxy_headers={"x-forwarded-for", "x-forwarded-proto"},
            clear_untrusted_proxy_headers=True,
        )
