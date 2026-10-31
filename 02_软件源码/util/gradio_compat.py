"""Local compatibility for Gradio's schemas and Windows upload completion.

Pydantic can emit additionalProperties=True in file component metadata. Gradio
4.44's client assumes every child schema is a dict and crashes even on GET /.
The schema shim leaves request validation unchanged. Upload middleware delays
the upload response until Gradio's background cache move has completed.
Upstream issue: https://github.com/gradio-app/gradio/issues/10792
"""

from functools import wraps

import gradio as gr
from starlette.middleware import Middleware


class CompleteUploadsBeforeResponse:
    """Finish Gradio's background upload move before advertising its path.

    On Windows a repeated upload cannot rename over an existing cache file.
    Gradio 4.44 then moves it in a response background task. Buffer only the
    small /upload response so a following callback never opens a partial file.
    Request bodies, SSE, and other responses retain their normal streaming.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or not scope.get("path", "").endswith("/upload"):
            return await self.app(scope, receive, send)
        response = []

        async def buffer(message):
            response.append(message)

        await self.app(scope, receive, buffer)
        for message in response:
            await send(message)


class WorkflowBlocks(gr.Blocks):
    """Apply upload readiness to every server launched from this UI."""

    @wraps(gr.Blocks.launch)
    def launch(self, *args, app_kwargs=None, **kwargs):
        options = dict(app_kwargs or {})
        options["middleware"] = [Middleware(CompleteUploadsBeforeResponse), *options.get("middleware", [])]
        return super().launch(*args, app_kwargs=options, **kwargs)


def support_boolean_schemas():
    from gradio_client import utils

    original = utils._json_schema_to_python_type
    try:
        original(True, {})
    except TypeError:
        pass
    else:
        return  # Already supported by the installed client (or this shim).

    @wraps(original)
    def convert(schema, defs=None):
        if isinstance(schema, bool):
            return "Any" if schema else "Never"
        return original(schema, defs)

    utils._json_schema_to_python_type = convert
