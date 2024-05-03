from typing import Any, AsyncGenerator, Optional, Union

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI

from approaches.approach import Approach, ThoughtStep
from core.authentication import AuthenticationHelper
from core.messagebuilder import MessageBuilder


class RetrieveThenReadApproach(Approach):
    """
    Implementação simples de recuperação e leitura, usando as APIs de IA Search e OpenAI diretamente. Primeiro, ele recupera
    os principais documentos da pesquisa, depois constrói um prompt com eles e, em seguida, usa o OpenAI para gerar uma conclusão
    (resposta) com esse prompt.
    """

    system_chat_template = (
        "Você é um assistente inteligente ajudando os funcionários da Atra com perguntas sobre um conjunto de contratos legais públicos."
        + "Use 'você' para se referir ao indivíduo que faz as perguntas, mesmo que elas sejam feitas com 'eu'."
        + "Responda à seguinte pergunta usando apenas os dados fornecidos nas fontes abaixo."
        + "Para informações tabulares, retorne-as como uma tabela HTML. Não retorne no formato markdown."
        + "Cada fonte tem um nome seguido por dois pontos e a informação real, sempre inclua o nome da fonte para cada fato que você usar na resposta."
        + "Se você não puder responder usando as fontes abaixo, diga que não sabe ou que não pode responder essa questão no momento."
        + "Se a pergunta do usuário não esitver em Português, traduza-a para o Português. Forneça as respostas apenas em Português."
    )

    # shots/sample conversation
    question = """
        'Qual é o valor médio desses contratos?'

        Fontes:
        contrato1.pdf: Este contrato, válido de 30/11/2023 a 30/11/2024, é para serviços relacionados ao recebimento, armazenamento e disposição final de pneus inutilizáveis em Sapezal-MT. O valor do contrato é de R$ 154.800,00.
        contrato2.pdf: Este contrato, com base na Lei Federal nº 14133, de 1º de abril, é para aquisição de itens a um custo total de R$ 469.899,99. Os pagamentos deste contrato serão feitos a partir de alocações orçamentárias específicas.
        contrato3.pdf: Este contrato, que não permite subcontratação, é para a prestação de serviços a um custo total de R$ 663.500,00. O custo inclui todas as despesas diretas e indiretas relacionadas à execução do contrato.
        contrato4.pdf: Este contrato, resultante da Dispensa de Licitação nº 27/2024, é para serviços especializados de elaboração de laudos, pareceres técnicos em perícias psiquiátricas e grafotécnicas. O valor total do contrato é de R$ 1.200,00. O pagamento será feito em até 30 dias após a emissão da fatura.
        """

    answer = "O valor médio dos contratos é de R$ 322.349,99. Isso é calculado adicionando os valores totais (R$ 154.800,00, R$ 469.899,99, R$ 663.500,00, R$ 1.200,00) e dividindo pelo número de contratos (4). Por favor, note que este é um cálculo simplificado e pode não levar em consideração outros fatores que poderiam afetar o valor médio dos contratos. É sempre uma boa ideia consultar um consultor financeiro ou contador para cálculos mais precisos."

    def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_model: str,
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_dimensions: int,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
    ):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller

    async def run(
        self,
        messages: list[dict],
        stream: bool = False,  # Stream is not used in this approach
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> Union[dict[str, Any], AsyncGenerator[dict[str, Any], None]]:
        q = messages[-1]["content"]
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = overrides.get("semantic_ranker") and has_text

        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top", 3)
        minimum_search_score = overrides.get("minimum_search_score", 0.030)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 3.0)
        filter = self.build_filter(overrides, auth_claims)
        # If retrieval mode includes vectors, compute an embedding for the query
        vectors: list[VectorQuery] = []
        if has_vector:
            vectors.append(await self.compute_text_embedding(q))

        # Only keep the text query if the retrieval mode uses text, otherwise drop it
        query_text = q if has_text else None

        results = await self.search(
            top,
            query_text,
            filter,
            vectors,
            use_semantic_ranker,
            use_semantic_captions,
            minimum_search_score,
            minimum_reranker_score,
        )

        user_content = [q]

        template = overrides.get("prompt_template", self.system_chat_template)
        model = self.chatgpt_model
        message_builder = MessageBuilder(template, model)

        # Process results
        sources_content = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)

        # Append user message
        content = "\n".join(sources_content)
        user_content = q + "\n" + f"Sources:\n {content}"
        message_builder.insert_message("user", user_content)
        message_builder.insert_message("assistant", self.answer)
        message_builder.insert_message("user", self.question)
        updated_messages = message_builder.messages
        chat_completion = (
            await self.openai_client.chat.completions.create(
                # Azure OpenAI takes the deployment name as the model name
                model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
                messages=updated_messages,
                temperature=overrides.get("temperature", 0.3),
                max_tokens=1024,
                n=1,
            )
        ).model_dump()

        data_points = {"text": sources_content}
        extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Search using user query",
                    query_text,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "top": top,
                        "filter": filter,
                        "has_vector": has_vector,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
                ThoughtStep(
                    "Prompt to generate answer",
                    [str(message) for message in updated_messages],
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
                ),
            ],
        }

        chat_completion["choices"][0]["context"] = extra_info
        chat_completion["choices"][0]["session_state"] = session_state
        return chat_completion
