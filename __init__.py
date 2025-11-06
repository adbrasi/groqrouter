import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


class DualAPILLMNode:
    """
    ComfyUI custom node that routes chat completion requests between Groq and OpenRouter
    with robust JSON parsing and defensive error handling.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_provider": (["groq", "openrouter"], {"default": "groq"}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are a helpful assistant. Always respond with valid JSON format.",
                    },
                ),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": ("STRING", {"multiline": False, "default": "llama-3.3-70b-versatile"}),
            },
            "optional": {
                "custom_parameters": ("STRING", {"multiline": True, "default": "{}"}),
                "timeout": ("INT", {"default": 60, "min": 10, "max": 300}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 10}),
            },
        }

    RETURN_TYPES = ("STRING",) * 10
    RETURN_NAMES = (
        "raw_response",
        "json_response",
        "value_1",
        "value_2",
        "value_3",
        "value_4",
        "value_5",
        "value_6",
        "value_7",
        "status",
    )
    FUNCTION = "execute_api_call"
    CATEGORY = "LLM/API"

    def parse_json_from_response(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to extract JSON objects from model responses using multiple strategies.
        """
        if not text:
            return None

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        json_blocks = re.findall(r"```json\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
        for block in json_blocks:
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue

        code_blocks = re.findall(r"```\s*\n(.*?)\n```", text, re.DOTALL)
        for block in code_blocks:
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue

        json_patterns = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        for pattern in json_patterns:
            try:
                return json.loads(pattern.strip())
            except json.JSONDecodeError:
                continue

        array_patterns = re.findall(r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]", text, re.DOTALL)
        for pattern in array_patterns:
            try:
                return json.loads(pattern.strip())
            except json.JSONDecodeError:
                continue

        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("{") or line.strip().startswith("["):
                for j in range(len(lines) - 1, i - 1, -1):
                    if lines[j].strip().endswith("}") or lines[j].strip().endswith("]"):
                        potential_json = "\n".join(lines[i : j + 1])
                        try:
                            return json.loads(potential_json.strip())
                        except json.JSONDecodeError:
                            continue

        return None

    def _coerce_prompt_sequence(self, prompt_input: Any) -> List[Any]:
        if prompt_input is None:
            return []
        if isinstance(prompt_input, (list, tuple)):
            return list(prompt_input)
        if isinstance(prompt_input, dict):
            return [prompt_input]
        if isinstance(prompt_input, str):
            stripped = prompt_input.strip()
            if not stripped:
                return []
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass
            return [prompt_input]
        return [prompt_input]

    def _stringify_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return str(value)

    def extract_value_strings(self, data: Any, limit: int = 7) -> List[str]:
        values: List[str] = []

        def add_value(value: Any) -> None:
            if len(values) >= limit:
                return

            if isinstance(value, (str, int, float, bool)) or value is None:
                values.append(self._stringify_value(value))
                return

            if isinstance(value, list):
                for item in value:
                    if len(values) >= limit:
                        break
                    add_value(item)
                return

            if isinstance(value, dict):
                for sub_key in value:
                    if len(values) >= limit:
                        break
                    add_value(value[sub_key])
                return

            values.append(self._stringify_value(value))

        add_value(data)
        return values

    def sanitize_custom_parameters(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {}
        sanitized: Dict[str, Any] = {}
        for key, value in params.items():
            if value is None:
                continue
            sanitized[str(key)] = value
        return sanitized

    def prepare_messages(
        self, system_prompt: Any, user_prompt: Any, extra_messages: Optional[Any] = None
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []

        system_prompt_text = ""
        if isinstance(system_prompt, str):
            system_prompt_text = system_prompt.strip()
        elif system_prompt is not None:
            system_prompt_text = self._stringify_value(system_prompt).strip()

        if system_prompt_text:
            messages.append({"role": "system", "content": system_prompt_text})

        combined_sequence: List[Any] = self._coerce_prompt_sequence(user_prompt)
        if extra_messages is not None:
            combined_sequence.extend(self._coerce_prompt_sequence(extra_messages))

        if not combined_sequence:
            raise ValueError("User prompt não fornecido")

        for entry in combined_sequence:
            if isinstance(entry, dict) and "content" in entry:
                role = entry.get("role", "user")
                content = entry.get("content", "")
                messages.append({"role": str(role), "content": self._stringify_value(content)})
            else:
                messages.append({"role": "user", "content": self._stringify_value(entry)})

        return messages

    def _finalize_outputs(
        self, raw_response: Any, json_response: Any, values: Optional[List[str]], status: Dict[str, Any]
    ) -> Tuple[str, str, str, str, str, str, str, str, str, str]:
        value_slots = [""] * 7
        if values:
            for index, value in enumerate(values[:7]):
                value_slots[index] = value if value is not None else ""

        raw_text = self._stringify_value(raw_response)
        json_text = self._stringify_value(json_response)

        try:
            status_text = json.dumps(status, indent=2)
        except (TypeError, ValueError):
            status_text = json.dumps(
                {
                    "status": "error",
                    "error": "Falha ao serializar status",
                    "raw_status": self._stringify_value(status),
                },
                indent=2,
            )

        return (
            raw_text or "",
            json_text or "",
            value_slots[0],
            value_slots[1],
            value_slots[2],
            value_slots[3],
            value_slots[4],
            value_slots[5],
            value_slots[6],
            status_text,
        )

    def _extract_error_detail(self, response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            error_obj = payload.get("error")
            if isinstance(error_obj, dict):
                message = error_obj.get("message") or error_obj.get("code") or ""
                if message:
                    return message
                return json.dumps(error_obj)
            if error_obj:
                return str(error_obj)
            return json.dumps(payload)

        text = (response.text or "").strip()
        return text[:500]

    def build_request_payload(
        self, provider: str, model: str, messages: List[Dict[str, str]], custom_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if not custom_params:
            return payload, [], {}

        applied: List[str] = []
        ignored: Dict[str, Any] = {}

        base_params = {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "top_a",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "max_tokens",
            "stop",
            "stream",
            "n",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "response_format",
            "seed",
            "tools",
            "tool_choice",
            "functions",
            "function_call",
            "metadata",
            "user",
            "reasoning_effort",
            "parallel_tool_calls",
        }
        if provider == "groq":
            base_params.update({"safety", "structured_output"})

        for key, value in custom_params.items():
            if key == "messages":
                continue
            if key in base_params:
                payload[key] = value
                applied.append(key)
            elif provider == "openrouter":
                payload[key] = value
                applied.append(key)
            else:
                ignored[key] = value

        return payload, applied, ignored

    def make_api_request(
        self, provider: str, api_key: str, payload: Dict[str, Any], timeout: int, max_retries: int
    ) -> Tuple[str, Dict[str, Any]]:
        if provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        elif provider == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://comfyui-custom-node",
                "X-Title": "ComfyUI Custom LLM Node",
            }
        else:
            return "", {
                "status": "error",
                "provider": provider,
                "model": payload.get("model", "unknown"),
                "error": f"Provider '{provider}' não suportado",
            }

        try:
            request_payload = json.loads(json.dumps(payload))
        except (TypeError, ValueError) as serialization_error:
            return "", {
                "status": "error",
                "provider": provider,
                "model": payload.get("model", "unknown"),
                "error": f"Payload não serializável: {serialization_error}",
            }

        last_error = ""

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=request_payload,
                    timeout=timeout,
                )

                if response.status_code == 200:
                    try:
                        response_data = response.json()
                    except ValueError:
                        return "", {
                            "status": "error",
                            "provider": provider,
                            "model": payload.get("model", "unknown"),
                            "error": "Resposta JSON inválida do provedor",
                            "attempt": attempt + 1,
                        }

                    choices = response_data.get("choices") or []
                    if choices:
                        choice = choices[0]
                        message_content = choice.get("message", {}).get("content", "") or ""
                        status_info: Dict[str, Any] = {
                            "status": "success",
                            "provider": provider,
                            "model": response_data.get("model", payload.get("model", "unknown")),
                            "tokens_used": response_data.get("usage", {}),
                            "response_time": response.elapsed.total_seconds(),
                            "attempt": attempt + 1,
                        }
                        request_id = response.headers.get("x-request-id") or response.headers.get("X-Request-Id")
                        if request_id:
                            status_info["request_id"] = request_id
                        if response_data.get("id"):
                            status_info["response_id"] = response_data["id"]
                        finish_reason = choice.get("finish_reason")
                        if finish_reason:
                            status_info["finish_reason"] = finish_reason
                        return message_content, status_info

                    last_error = "Resposta inválida do provedor: campo 'choices' ausente ou vazio"
                else:
                    detail = self._extract_error_detail(response)
                    last_error = f"HTTP {response.status_code}: {detail}"

            except requests.exceptions.Timeout:
                last_error = f"Timeout na tentativa {attempt + 1}/{max_retries}"
            except requests.exceptions.RequestException as exc:
                last_error = f"Erro de conexão: {str(exc)}"
            except Exception as exc:
                last_error = f"Erro inesperado: {str(exc)}"

            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 10)
                time.sleep(wait_time)

        return "", {
            "status": "error",
            "provider": provider,
            "model": payload.get("model", "unknown"),
            "error": last_error,
            "attempts": max_retries,
        }

    def execute_api_call(
        self,
        api_provider: str,
        api_key: str,
        system_prompt: str,
        user_prompt: Any,
        model: str,
        custom_parameters: Any = "{}",
        timeout: int = 60,
        max_retries: int = 3,
    ) -> Tuple[str, str, str, str, str, str, str, str, str, str]:
        try:
            if not str(api_key or "").strip():
                status = {
                    "status": "error",
                    "error": "API key não fornecida",
                    "provider": api_provider,
                }
                return self._finalize_outputs("", "", [], status)

            custom_params_raw: Any = {}
            if isinstance(custom_parameters, str):
                if custom_parameters.strip():
                    try:
                        custom_params_raw = json.loads(custom_parameters)
                    except json.JSONDecodeError as decode_error:
                        status = {
                            "status": "error",
                            "error": f"Parâmetros customizados inválidos: {decode_error}",
                            "provider": api_provider,
                        }
                        return self._finalize_outputs("", "", [], status)
            elif isinstance(custom_parameters, dict):
                custom_params_raw = custom_parameters

            if custom_params_raw and not isinstance(custom_params_raw, dict):
                status = {
                    "status": "error",
                    "error": "Parâmetros customizados devem ser um objeto JSON",
                    "provider": api_provider,
                }
                return self._finalize_outputs("", "", [], status)

            custom_params = self.sanitize_custom_parameters(custom_params_raw)
            custom_params = dict(custom_params)

            extra_messages = custom_params.pop("messages", None)

            try:
                messages = self.prepare_messages(system_prompt, user_prompt, extra_messages)
            except ValueError as missing_prompt:
                status = {
                    "status": "error",
                    "error": str(missing_prompt),
                    "provider": api_provider,
                }
                return self._finalize_outputs("", "", [], status)

            payload, applied_params, ignored_params = self.build_request_payload(
                api_provider, model, messages, custom_params
            )

            response_content, status_data = self.make_api_request(
                api_provider, api_key, payload, timeout, max_retries
            )

            status_data = status_data or {}
            status_data.setdefault("provider", api_provider)
            status_data.setdefault("model", payload.get("model", model))
            status_data["message_count"] = len(messages)
            if applied_params:
                status_data["applied_params"] = applied_params
            if ignored_params:
                status_data["ignored_params"] = ignored_params

            if not response_content:
                return self._finalize_outputs("", "", [], status_data)

            parsed_json = self.parse_json_from_response(response_content)
            json_response = ""
            extracted_values: List[str] = []

            if parsed_json is not None:
                try:
                    json_response = json.dumps(parsed_json, indent=2)
                except (TypeError, ValueError):
                    json_response = str(parsed_json)
                extracted_values = self.extract_value_strings(parsed_json)
            else:
                extracted_values = [self._stringify_value(response_content)]

            return self._finalize_outputs(response_content, json_response, extracted_values, status_data)

        except Exception as exc:
            status = {
                "status": "error",
                "error": f"Erro interno: {str(exc)}",
                "provider": api_provider,
            }
            return self._finalize_outputs("", "", [], status)


NODE_CLASS_MAPPINGS = {
    "DualAPILLMNode": DualAPILLMNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DualAPILLMNode": "Groq/OpenRouter",
}
