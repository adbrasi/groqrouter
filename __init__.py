import requests
import json
import re
import time
from typing import Dict, Any, Tuple, List, Optional, Union

class DualAPILLMNode:
    """
    Node customizado para ComfyUI que permite chamadas flexíveis para APIs Groq e OpenRouter
    com configurações personalizáveis e parsing robusto de JSON
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_provider": (["groq", "openrouter"], {"default": "groq"}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant. Always respond with valid JSON format."}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": ("STRING", {"multiline": False, "default": "llama-3.3-70b-versatile"}),
            },
            "optional": {
                "custom_parameters": ("STRING", {"multiline": True, "default": "{}"}),
                "timeout": ("INT", {"default": 60, "min": 10, "max": 300}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 10}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_1", "output_2", "output_3", "output_4", "status")
    FUNCTION = "execute_api_call"
    CATEGORY = "LLM/API"
    
    def parse_json_from_response(self, text: str) -> Optional[Dict]:
        """
        Parser robusto para extrair JSON da resposta da LLM
        Tenta múltiplas estratégias para encontrar JSON válido
        """
        if not text:
            return None
            
        # Estratégia 1: Tentar parsear o texto inteiro
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Estratégia 2: Procurar por blocos JSON entre ```json e ```
        json_blocks = re.findall(r'```json\s*\n(.*?)\n```', text, re.DOTALL | re.IGNORECASE)
        for block in json_blocks:
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue
        
        # Estratégia 3: Procurar por blocos JSON entre ``` e ```
        code_blocks = re.findall(r'```\s*\n(.*?)\n```', text, re.DOTALL)
        for block in code_blocks:
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue
        
        # Estratégia 4: Procurar por padrões { ... } no texto
        json_patterns = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        for pattern in json_patterns:
            try:
                return json.loads(pattern.strip())
            except json.JSONDecodeError:
                continue
        
        # Estratégia 5: Procurar por arrays [ ... ]
        array_patterns = re.findall(r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', text, re.DOTALL)
        for pattern in array_patterns:
            try:
                return json.loads(pattern.strip())
            except json.JSONDecodeError:
                continue
        
        # Estratégia 6: Tentar remover texto antes e depois de possível JSON
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('{') or line.strip().startswith('['):
                for j in range(len(lines) - 1, i - 1, -1):
                    if lines[j].strip().endswith('}') or lines[j].strip().endswith(']'):
                        potential_json = '\n'.join(lines[i:j+1])
                        try:
                            return json.loads(potential_json.strip())
                        except json.JSONDecodeError:
                            continue
        
        return None
    
    def build_request_payload(self, system_prompt: str, user_prompt: str, model: str, custom_params: Dict) -> Dict:
        """
        Constrói o payload da requisição com parâmetros personalizados
        """
        # Payload base seguindo formato OpenAI
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        
        # Adicionar parâmetros customizados
        if custom_params:
            # Parâmetros de sampling comuns
            sampling_params = [
                'temperature', 'top_p', 'top_k', 'frequency_penalty', 
                'presence_penalty', 'repetition_penalty', 'min_p', 'top_a',
                'max_tokens', 'seed', 'stop', 'logit_bias', 'logprobs', 
                'top_logprobs', 'response_format', 'tool_choice', 'tools',
                'reasoning_effort'  # Para modelos com reasoning
            ]
            
            for param in sampling_params:
                if param in custom_params:
                    payload[param] = custom_params[param]
            
            # Parâmetros específicos que podem ir no extra_body (OpenRouter)
            extra_body_params = {}
            for key, value in custom_params.items():
                if key not in sampling_params and key not in payload:
                    extra_body_params[key] = value
            
            if extra_body_params:
                payload['extra_body'] = extra_body_params
        
        return payload
    
    def make_api_request(self, provider: str, api_key: str, payload: Dict, timeout: int, max_retries: int) -> Tuple[str, str]:
        """
        Faz a requisição para a API com retry automático
        """
        # Configurar URL e headers baseado no provider
        if provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        elif provider == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://comfyui-custom-node",
                "X-Title": "ComfyUI Custom LLM Node"
            }
            # Para OpenRouter, mover extra_body para o nível principal
            if 'extra_body' in payload:
                extra_body = payload.pop('extra_body')
                payload.update(extra_body)
        else:
            return "", f"Erro: Provider '{provider}' não suportado"
        
        # Fazer requisição com retry
        last_error = ""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=payload, 
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        content = response_data['choices'][0]['message']['content']
                        status_info = {
                            "status": "success",
                            "provider": provider,
                            "model": payload.get('model', 'unknown'),
                            "tokens_used": response_data.get('usage', {}),
                            "response_time": response.elapsed.total_seconds()
                        }
                        return content, json.dumps(status_info, indent=2)
                    else:
                        last_error = f"Resposta inválida: {response.text}"
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    
            except requests.exceptions.Timeout:
                last_error = f"Timeout na tentativa {attempt + 1}/{max_retries}"
            except requests.exceptions.RequestException as e:
                last_error = f"Erro de conexão: {str(e)}"
            except Exception as e:
                last_error = f"Erro inesperado: {str(e)}"
            
            # Aguardar antes da próxima tentativa (backoff exponencial)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        
        # Se chegou aqui, todas as tentativas falharam
        error_status = {
            "status": "error",
            "provider": provider,
            "model": payload.get('model', 'unknown'),
            "error": last_error,
            "attempts": max_retries
        }
        return "", json.dumps(error_status, indent=2)
    
    def execute_api_call(self, api_provider: str, api_key: str, system_prompt: str, 
                        user_prompt: str, model: str, custom_parameters: str = "{}", 
                        timeout: int = 60, max_retries: int = 3) -> Tuple[str, str, str, str, str]:
        """
        Executa a chamada da API e processa a resposta
        """
        try:
            # Validar inputs obrigatórios
            if not api_key.strip():
                return "", "", "", "", '{"status": "error", "error": "API key não fornecida"}'
            
            if not user_prompt.strip():
                return "", "", "", "", '{"status": "error", "error": "User prompt não fornecido"}'
            
            # Parsear parâmetros customizados
            try:
                custom_params = json.loads(custom_parameters) if custom_parameters.strip() else {}
            except json.JSONDecodeError as e:
                return "", "", "", "", f'{{"status": "error", "error": "Parâmetros customizados inválidos: {str(e)}"}}'
            
            # Construir payload da requisição
            payload = self.build_request_payload(system_prompt, user_prompt, model, custom_params)
            
            # Fazer requisição para API
            response_content, status = self.make_api_request(
                api_provider, api_key, payload, timeout, max_retries
            )
            
            # Se houve erro na requisição
            if not response_content:
                return "", "", "", "", status
            
            # Tentar extrair JSON da resposta
            parsed_json = self.parse_json_from_response(response_content)
            
            if parsed_json is None:
                # Se não conseguiu parsear JSON, retornar resposta bruta no primeiro output
                return response_content, "", "", "", status
            
            # Processar JSON parseado
            outputs = ["", "", "", ""]
            
            if isinstance(parsed_json, dict):
                # Se é um dicionário, pegar os valores em ordem
                values = list(parsed_json.values())
                for i, value in enumerate(values[:4]):
                    outputs[i] = str(value) if value is not None else ""
            elif isinstance(parsed_json, list):
                # Se é uma lista, pegar os itens em ordem
                for i, item in enumerate(parsed_json[:4]):
                    outputs[i] = str(item) if item is not None else ""
            else:
                # Se é outro tipo, colocar tudo no primeiro output
                outputs[0] = str(parsed_json)
            
            return outputs[0], outputs[1], outputs[2], outputs[3], status
            
        except Exception as e:
            error_status = {
                "status": "error",
                "error": f"Erro interno: {str(e)}",
                "provider": api_provider
            }
            return "", "", "", "", json.dumps(error_status, indent=2)

# Registrar o node no ComfyUI
NODE_CLASS_MAPPINGS = {
    "DualAPILLMNode": DualAPILLMNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DualAPILLMNode": "Groq/OpenRouter)"
}