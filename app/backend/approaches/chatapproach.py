import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Optional, Union

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
)

from approaches.approach import Approach
from core.messagebuilder import MessageBuilder


class ChatApproach(Approach, ABC):
    # Chat roles
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    query_prompt_few_shots = [
        {"role": USER, "content": "Qual foi o motivo da apelação no caso de locação de aluguel?"},
        {"role": ASSISTANT, "content": "Identifique o motivo da apelação no caso de locação de aluguel"},
        {"role": USER, "content": "Quais foram os argumentos apresentados pelo apelante no caso trabalhista?"},
        {"role": ASSISTANT, "content": "Liste os argumentos apresentados pelo apelante no caso trabalhista"},
        {"role": USER, "content": "Qual foi a decisão do juiz em primeira instância no caso de locação de aluguel?"},
        {"role": ASSISTANT, "content": "Descreva a decisão do juiz em primeira instância no caso de locação de aluguel"},
        {"role": USER, "content": "Quais foram as provas apresentadas no caso trabalhista?"},
        {"role": ASSISTANT, "content": "Identifique as provas apresentadas no caso trabalhista"},
        {"role": USER, "content": "Qual foi o resultado da apelação no caso de locação de aluguel?"},
        {"role": ASSISTANT, "content": "Informe o resultado da apelação no caso de locação de aluguel"},
        {"role": USER, "content": "Qual é o procedimento para entrar com uma ação de despejo por falta de pagamento?"},
        {"role": ASSISTANT, "content": "Explique o procedimento para entrar com uma ação de despejo por falta de pagamento"},
        {"role": USER, "content": "Quais são os requisitos para caracterizar uma rescisão indireta do contrato de trabalho?"},
        {"role": ASSISTANT, "content": "Descreva os requisitos para caracterizar uma rescisão indireta do contrato de trabalho"},
        {"role": USER, "content": "Quais são os prazos para interpor recurso após uma sentença judicial?"},
        {"role": ASSISTANT, "content": "Informe os prazos para interpor recurso após uma sentença judicial"},
        {"role": USER, "content": "Como é calculada a indenização por danos morais em um processo trabalhista?"},
        {"role": ASSISTANT, "content": "Explique como é calculada a indenização por danos morais em um processo trabalhista"},
        {"role": USER, "content": "Quais são os documentos necessários para instruir uma ação de cobrança de aluguel em atraso?"},
        {"role": ASSISTANT, "content": "Liste os documentos necessários para instruir uma ação de cobrança de aluguel em atraso"},
        {"role": USER, "content": "Qual é a diferença entre um contrato de locação residencial e um contrato de locação comercial?"},
        {"role": ASSISTANT, "content": "Explique a diferença entre um contrato de locação residencial e um contrato de locação comercial"},
        {"role": USER, "content": "Quais são as formas de garantia de um contrato de locação?"},
        {"role": ASSISTANT, "content": "Identifique as formas de garantia de um contrato de locação"},
    ]
    NO_RESPONSE = "0"

    follow_up_questions_prompt_content = """Gere 3 perguntas de acompanhamento breves que o usuário provavelmente faria a seguir.
    Coloque as perguntas de acompanhamento entre colchetes duplos. Exemplo:
    <<Qual é o prazo para notificar o locador sobre uma reparação?>>
    <<Como faço para calcular a rescisão do contrato de trabalho?>>
    <<Quais são os documentos necessários para iniciar uma ação trabalhista?>>
    Não repita perguntas que já foram feitas.
    Certifique-se de que a última pergunta termine com ">>".
    """


    query_prompt_template = """Abaixo está um histórico da conversa até agora e uma nova pergunta feita pelo usuário que precisa ser respondida buscando em uma base de conhecimento.
    Você tem acesso a um índice de pesquisa com centenas de documentos.
    Gere uma consulta de pesquisa com base na conversa e na nova pergunta.
    Não inclua nomes de arquivos citados e nomes de documentos, por exemplo info.txt ou doc.pdf, nos termos da consulta de pesquisa.
    Não inclua nenhum texto dentro de [] ou <<>> nos termos da consulta de pesquisa.
    Não inclua caracteres especiais como '+'.
    Se não puder gerar uma consulta de pesquisa, retorne apenas o número 0.
    """


    @property
    @abstractmethod
    def system_message_chat_conversation(self) -> str:
        pass

    @abstractmethod
    async def run_until_final_call(self, history, overrides, auth_claims, should_stream) -> tuple:
        pass

    def get_system_prompt(self, override_prompt: Optional[str], follow_up_questions_prompt: str) -> str:
        if override_prompt is None:
            return self.system_message_chat_conversation.format(
                injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt
            )
        elif override_prompt.startswith(">>>"):
            return self.system_message_chat_conversation.format(
                injected_prompt=override_prompt[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt
            )
        else:
            return override_prompt.format(follow_up_questions_prompt=follow_up_questions_prompt)

    def get_search_query(self, chat_completion: ChatCompletion, user_query: str):
        response_message = chat_completion.choices[0].message

        if response_message.tool_calls:
            for tool in response_message.tool_calls:
                if tool.type != "function":
                    continue
                function = tool.function
                if function.name == "search_sources":
                    arg = json.loads(function.arguments)
                    search_query = arg.get("search_query", self.NO_RESPONSE)
                    if search_query != self.NO_RESPONSE:
                        return search_query
        elif query_text := response_message.content:
            if query_text.strip() != self.NO_RESPONSE:
                return query_text
        return user_query

    def extract_followup_questions(self, content: str):
        return content.split("<<")[0], re.findall(r"<<([^>>]+)>>", content)

    def get_messages_from_history(
        self,
        system_prompt: str,
        model_id: str,
        history: list[dict[str, str]],
        user_content: Union[str, list[ChatCompletionContentPartParam]],
        max_tokens: int,
        few_shots=[],
    ) -> list[ChatCompletionMessageParam]:
        message_builder = MessageBuilder(system_prompt, model_id)

        # Add examples to show the chat what responses we want. It will try to mimic any responses and make sure they match the rules laid out in the system message.
        for shot in reversed(few_shots):
            message_builder.insert_message(shot.get("role"), shot.get("content"))

        append_index = len(few_shots) + 1

        message_builder.insert_message(self.USER, user_content, index=append_index)

        total_token_count = 0
        for existing_message in message_builder.messages:
            total_token_count += message_builder.count_tokens_for_message(existing_message)

        newest_to_oldest = list(reversed(history[:-1]))
        for message in newest_to_oldest:
            potential_message_count = message_builder.count_tokens_for_message(message)
            if (total_token_count + potential_message_count) > max_tokens:
                logging.info("Reached max tokens of %d, history will be truncated", max_tokens)
                break
            message_builder.insert_message(message["role"], message["content"], index=append_index)
            total_token_count += potential_message_count
        return message_builder.messages

    async def run_without_streaming(
        self,
        history: list[dict[str, str]],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> dict[str, Any]:
        extra_info, chat_coroutine = await self.run_until_final_call(
            history, overrides, auth_claims, should_stream=False
        )
        chat_completion_response: ChatCompletion = await chat_coroutine
        chat_resp = chat_completion_response.model_dump()  # Convert to dict to make it JSON serializable
        chat_resp["choices"][0]["context"] = extra_info
        if overrides.get("suggest_followup_questions"):
            content, followup_questions = self.extract_followup_questions(chat_resp["choices"][0]["message"]["content"])
            chat_resp["choices"][0]["message"]["content"] = content
            chat_resp["choices"][0]["context"]["followup_questions"] = followup_questions
        chat_resp["choices"][0]["session_state"] = session_state
        return chat_resp

    async def run_with_streaming(
        self,
        history: list[dict[str, str]],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> AsyncGenerator[dict, None]:
        extra_info, chat_coroutine = await self.run_until_final_call(
            history, overrides, auth_claims, should_stream=True
        )
        yield {
            "choices": [
                {
                    "delta": {"role": self.ASSISTANT},
                    "context": extra_info,
                    "session_state": session_state,
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "object": "chat.completion.chunk",
        }

        followup_questions_started = False
        followup_content = ""
        async for event_chunk in await chat_coroutine:
            # "2023-07-01-preview" API version has a bug where first response has empty choices
            event = event_chunk.model_dump()  # Convert pydantic model to dict
            if event["choices"]:
                # if event contains << and not >>, it is start of follow-up question, truncate
                content = event["choices"][0]["delta"].get("content")
                content = content or ""  # content may either not exist in delta, or explicitly be None
                if overrides.get("suggest_followup_questions") and "<<" in content:
                    followup_questions_started = True
                    earlier_content = content[: content.index("<<")]
                    if earlier_content:
                        event["choices"][0]["delta"]["content"] = earlier_content
                        yield event
                    followup_content += content[content.index("<<") :]
                elif followup_questions_started:
                    followup_content += content
                else:
                    yield event
        if followup_content:
            _, followup_questions = self.extract_followup_questions(followup_content)
            yield {
                "choices": [
                    {
                        "delta": {"role": self.ASSISTANT},
                        "context": {"followup_questions": followup_questions},
                        "finish_reason": None,
                        "index": 0,
                    }
                ],
                "object": "chat.completion.chunk",
            }

    async def run(
        self, messages: list[dict], stream: bool = False, session_state: Any = None, context: dict[str, Any] = {}
    ) -> Union[dict[str, Any], AsyncGenerator[dict[str, Any], None]]:
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})

        if stream is False:
            return await self.run_without_streaming(messages, overrides, auth_claims, session_state)
        else:
            return self.run_with_streaming(messages, overrides, auth_claims, session_state)