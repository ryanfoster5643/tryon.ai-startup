import asyncio
import base64
import io
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from PIL import Image


class NanoBananaProcessor:
    """
    Lightweight try-on processor for vision-agents pipeline.

    Design:
    - `process(frame)` is called on each incoming video frame from GetStream.
    - We cache the latest frame for snapshot-based try-on generation.
    - `generate_tryon()` runs on demand (LLM function or external trigger) to avoid
      expensive per-frame generation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash-image",
        output_dir: str | Path = "outputs",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.call: Any | None = None
        self.merchandise_image: str | None = None
        self.pose_image: str | None = None
        self.latest_frame: Any | None = None

        self._inflight_lock = asyncio.Lock()

    def attach_call(self, call: Any) -> None:
        self.call = call

    async def process(self, frame: Any) -> Any:
        # Keep video pipeline real-time: store frame and return immediately.
        self.latest_frame = frame
        return frame

    async def set_merchandise(self, image_path_or_url: str) -> str:
        self.merchandise_image = image_path_or_url.strip()
        return self.merchandise_image

    async def set_pose_image(self, image_path_or_url: str) -> str:
        self.pose_image = image_path_or_url.strip()
        return self.pose_image

    async def clear_merchandise(self) -> None:
        self.merchandise_image = None

    async def clear_pose_image(self) -> None:
        self.pose_image = None

    async def generate_tryon(self) -> dict[str, Any]:
        start = time.perf_counter()

        if self.pose_image is None and self.latest_frame is None:
            return {
                "status": "error",
                "reason": "no_pose_or_frame_available",
                "message": "No captured pose image or video frame has been received yet.",
            }

        if not self.merchandise_image:
            return {
                "status": "error",
                "reason": "missing_merchandise",
                "message": "Please set a merchandise image first.",
            }

        if self._inflight_lock.locked():
            return {
                "status": "busy",
                "message": "A try-on generation is already in progress.",
            }

        async with self._inflight_lock:
            job_id = str(uuid.uuid4())
            result = await self._run_tryon(job_id)

            latency_ms = int((time.perf_counter() - start) * 1000)
            result["job_id"] = job_id
            result["latency_ms"] = latency_ms
            return result

    async def emit_result(self, result: dict[str, Any]) -> None:
        if self.call is None:
            return

        status = result.get("status", "unknown")
        payload = {
            "event": "mirror_update",
            "job_id": result.get("job_id"),
            "item_id": result.get("item_id"),
            "status": status,
            "image_url": result.get("image_url"),
            "source_image_url": self.merchandise_image,
            "latency_ms": result.get("latency_ms"),
            "reason": result.get("reason"),
            "message": result.get("message"),
            "error_message": result.get("message") if status != "success" else "",
        }

        try:
            maybe_coro = self.call.send_custom_event("mirror_update", payload)
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro
        except Exception:
            # Non-fatal: keep agent stable even if custom events are unsupported.
            pass

    @staticmethod
    def _is_url(value: str) -> bool:
        return value.startswith("http://") or value.startswith("https://")

    @staticmethod
    def _decode_maybe_base64(data: bytes | str) -> bytes:
        if isinstance(data, bytes):
            return data
        try:
            return base64.b64decode(data, validate=True)
        except Exception:
            return data.encode("utf-8")

    @staticmethod
    def _looks_like_image_bytes(data: bytes) -> bool:
        if not data:
            return False
        signatures = (
            b"\x89PNG\r\n\x1a\n",
            b"\xff\xd8\xff",
            b"GIF87a",
            b"GIF89a",
            b"RIFF",
            b"BM",
        )
        if any(data.startswith(sig) for sig in signatures):
            return True
        if data[:12].startswith(b"RIFF") and b"WEBP" in data[:16]:
            return True
        return False

    def _read_image_bytes_from_path_or_url(self, source: str) -> bytes:
        source = source.strip()

        if source.startswith("data:image/"):
            header, encoded = source.split(",", 1)
            if ";base64" in header:
                raw = base64.b64decode(encoded, validate=False)
            else:
                raw = urllib.parse.unquote_to_bytes(encoded)
            if not self._looks_like_image_bytes(raw):
                raise ValueError("Data URL did not contain a valid image payload.")
            return raw

        if self._is_url(source):
            request = urllib.request.Request(
                source,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; TryOnAI/1.0)",
                    "Accept": "image/*,*/*;q=0.8",
                },
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.read()
        return Path(source).read_bytes()

    @staticmethod
    def _format_source_fetch_error(
        source_label: str,
        source_value: str,
        exc: Exception,
    ) -> dict[str, str]:
        if isinstance(exc, urllib.error.HTTPError):
            hint = ""
            if exc.code == 403:
                hint = " Source URL is blocked (403). Use a local file path or publicly accessible URL."
            return {
                "status": "error",
                "reason": f"{source_label}_fetch_failed",
                "message": (
                    f"Failed to fetch {source_label.replace('_', ' ')} from '{source_value}': "
                    f"HTTP {exc.code} {exc.reason}.{hint}"
                ),
            }

        if isinstance(exc, urllib.error.URLError):
            return {
                "status": "error",
                "reason": f"{source_label}_fetch_failed",
                "message": (
                    f"Failed to fetch {source_label.replace('_', ' ')} from '{source_value}': "
                    f"{exc.reason}"
                ),
            }

        if isinstance(exc, FileNotFoundError):
            return {
                "status": "error",
                "reason": f"{source_label}_fetch_failed",
                "message": (
                    f"Failed to read {source_label.replace('_', ' ')} at '{source_value}': "
                    "file not found."
                ),
            }

        return {
            "status": "error",
            "reason": f"{source_label}_fetch_failed",
            "message": f"Failed to fetch {source_label.replace('_', ' ')}: {exc}",
        }

    def _classify_model_error(self, exc: Exception) -> dict[str, str]:
        text = str(exc)
        lower_text = text.lower()
        status_code = getattr(exc, "status_code", None)

        if status_code == 429 or "resource_exhausted" in lower_text or "quota exceeded" in lower_text:
            return {
                "status": "error",
                "reason": "model_quota_exhausted",
                "message": (
                    "Model quota exceeded. Check billing/quota limits and retry. "
                    f"Details: {text}"
                ),
            }

        if status_code == 403 or "forbidden" in lower_text or "permission denied" in lower_text:
            return {
                "status": "error",
                "reason": "model_access_forbidden",
                "message": (
                    "Model access is forbidden for this API key/project. "
                    "Verify model entitlement and API permissions. "
                    f"Details: {text}"
                ),
            }

        if "invalid argument" in lower_text or "not found" in lower_text or "unsupported" in lower_text:
            return {
                "status": "error",
                "reason": "invalid_model_or_request",
                "message": (
                    f"Invalid model/request for '{self.model}'. "
                    f"Details: {text}"
                ),
            }

        return {
            "status": "error",
            "reason": "tryon_generation_failed",
            "message": f"Try-on generation failed: {text}",
        }

    def _image_bytes_from_frame(self, frame: Any) -> bytes:
        if isinstance(frame, (bytes, bytearray, memoryview)):
            return bytes(frame)

        if isinstance(frame, Image.Image):
            image = frame.convert("RGB")
        elif hasattr(frame, "to_ndarray"):
            ndarray = frame.to_ndarray(format="rgb24")
            image = Image.fromarray(ndarray).convert("RGB")
        elif hasattr(frame, "__array_interface__") or hasattr(frame, "shape"):
            image = Image.fromarray(frame).convert("RGB")
        else:
            raise ValueError(f"Unsupported frame type for image conversion: {type(frame)}")

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _extract_inline_image_bytes(response: Any) -> bytes | None:
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                if not inline_data:
                    continue
                data = getattr(inline_data, "data", None)
                if data:
                    decoded = NanoBananaProcessor._decode_maybe_base64(data)
                    if decoded.startswith(b"\x89PNG") or decoded.startswith(b"\xff\xd8\xff"):
                        return decoded

        generated_images = getattr(response, "generated_images", None) or []
        for image_obj in generated_images:
            image = getattr(image_obj, "image", None)
            image_bytes = getattr(image, "image_bytes", None) if image else None
            if image_bytes:
                decoded = NanoBananaProcessor._decode_maybe_base64(image_bytes)
                if decoded.startswith(b"\x89PNG") or decoded.startswith(b"\xff\xd8\xff"):
                    return decoded
        return None

    @staticmethod
    def _to_browser_image_url(output_file: Path, image_bytes: bytes) -> str:
        """
        Prefer a URL path when output is written under try-on-react/public.
        Fall back to a data URL so frontend can still render without static hosting.
        """
        resolved = output_file.resolve()
        public_marker = "/try-on-react/public/"
        resolved_str = resolved.as_posix()
        marker_index = resolved_str.find(public_marker)
        if marker_index >= 0:
            relative = resolved_str[marker_index + len(public_marker):]
            return f"/{relative}"

        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    async def _run_tryon(self, job_id: str) -> dict[str, Any]:
        """
        Real try-on invocation via Google GenAI image model.

        Input A: captured pose image (or latest frame fallback)
        Input B: selected merchandise image
        Output: generated try-on image saved locally and returned as image_url
        """
        if not self.api_key:
            return {
                "status": "error",
                "reason": "missing_api_key",
                "message": "GOOGLE_API_KEY is not set.",
            }

        if self.pose_image is not None:
            pose_source = self.pose_image
            try:
                base_image_bytes = await asyncio.to_thread(
                    self._read_image_bytes_from_path_or_url, self.pose_image
                )
            except Exception as exc:
                return self._format_source_fetch_error("pose_image", self.pose_image, exc)
        else:
            pose_source = "latest_frame"
            try:
                base_image_bytes = await asyncio.to_thread(
                    self._image_bytes_from_frame, self.latest_frame
                )
            except Exception as exc:
                return {
                    "status": "error",
                    "reason": "pose_image_fetch_failed",
                    "message": f"Failed to convert latest frame into image bytes: {exc}",
                }

        try:
            garment_image_bytes = await asyncio.to_thread(
                self._read_image_bytes_from_path_or_url, self.merchandise_image
            )
        except Exception as exc:
            return self._format_source_fetch_error(
                "merchandise_image",
                self.merchandise_image,
                exc,
            )

        prompt = (
            "You are a virtual fitting room. You will receive two images:\n"
            "- IMAGE 1 (person): a photo of a person standing in front of a mirror or camera.\n"
            "- IMAGE 2 (garment): a clothing item to try on.\n\n"
            "Task: Generate a single photorealistic image of the person from IMAGE 1 wearing the garment from IMAGE 2.\n\n"
            "Rules:\n"
            "- Preserve the person's exact face, skin tone, hair, body shape, pose, and proportions from IMAGE 1.\n"
            "- Preserve the original background, lighting, and camera angle from IMAGE 1.\n"
            "- Keep all accessories (shoes, bags, jewelry) from IMAGE 1 unless hidden by the new garment.\n"
            "- Replace ONLY the clothing with the garment from IMAGE 2, fitting it naturally to the person's body.\n"
            "- Match the garment's color, pattern, texture, and style exactly as shown in IMAGE 2.\n"
            "- Do not change the person's identity, expression, or any other aspect of the scene.\n"
            "Output: one photorealistic edited image only."
        )

        def _mime_type(data: bytes) -> str:
            if data[:8] == b"\x89PNG\r\n\x1a\n":
                return "image/png"
            if data[:2] == b"\xff\xd8":
                return "image/jpeg"
            if data[:4] == b"RIFF" and b"WEBP" in data[:12]:
                return "image/webp"
            return "image/png"

        client = genai.Client(api_key=self.api_key)
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=base_image_bytes, mime_type=_mime_type(base_image_bytes)),
                    types.Part.from_bytes(data=garment_image_bytes, mime_type=_mime_type(garment_image_bytes)),
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
        except Exception as exc:
            return self._classify_model_error(exc)

        image_bytes = self._extract_inline_image_bytes(response)
        if not image_bytes:
            return {
                "status": "error",
                "reason": "no_image_in_model_response",
                "message": "Model response did not include an edited image.",
            }

        output_file = self.output_dir / f"tryon_{job_id}.png"
        output_file.write_bytes(image_bytes)
        browser_image_url = self._to_browser_image_url(output_file, image_bytes)

        return {
            "status": "success",
            "image_url": browser_image_url,
            "local_image_path": str(output_file),
            "message": f"Try-on image generated from {pose_source}.",
        }
