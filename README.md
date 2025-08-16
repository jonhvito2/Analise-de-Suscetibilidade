# Análise de Suscetibilidade (Streamlit)

Este app lê um Excel de resultados de suscetibilidade e mostra tabelas e gráficos interativos por antibiótico.

## Como rodar

1. Crie um ambiente e instale as dependências:
```
pip install -r requirements.txt
```

2. Inicie o app:
```
streamlit run app.py
```

3. No navegador, envie o arquivo Excel (o app espera cabeçalho na 2ª linha por padrão) e ajuste os filtros na barra lateral.

## Notas
- Detecta automaticamente as colunas "CÓDIGO UFPB" e "MALDI-TOF" (ou variações em caixa), considerando hifens especiais.
- Os antibióticos considerados inicialmente seguem a legenda do script original, mas você pode selecionar quais usar.
- O limite numérico (ex.: MA >= 180) separa "2025/atual" de "Anos anteriores".
- Baixe os resultados da tabela em CSV com o botão disponível em cada aba.
