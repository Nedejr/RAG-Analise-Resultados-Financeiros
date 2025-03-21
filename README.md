# Como Funciona?

Extrair o texto do PDF com os resultados financeiros.
Divide o texto em partes menores e gera embeddings para indexação com FAISS.
Cria um pipeline RAG para recuperar informações relevantes do PDF.
Usar um modelo de IA (GPT-4 da OpenAI) para analisar a situação financeira e dar uma recomendação.

## Saída Esperada
O modelo analisará os resultados financeiros e retornará algo como:

## Análise Financeira da Empresa:
Os resultados mostram um crescimento sólido da receita e uma boa margem de lucro. Com base nesses dados, parece uma boa ideia continuar investindo na empresa.
Caso os resultados sejam ruins:

## Melhorias Possíveis
Usar LlamaIndex ou ChromaDB em vez de FAISS para armazenamento vetorial.
Testar modelos open-source como Mistral ou Llama2 para evitar custos com OpenAI.
Implementar um frontend com Streamlit para análise visual interativa.
Se precisar de mais ajustes, me avise! 🚀