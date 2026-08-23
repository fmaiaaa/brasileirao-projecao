# Projeção Brasileirão 2026

App Streamlit para projetar a classificação final do Campeonato Brasileiro com base em resultados reais e modelos configuráveis.

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Atualizar resultados

**Recomendado:** edite a planilha Google Sheets (link no app):

https://docs.google.com/spreadsheets/d/1QkOIvRa9YinnOveOK4BkX4h_ZtGYRg1ZLzIfokbQh5I/edit

| Coluna     | Descrição                                      |
|------------|------------------------------------------------|
| Rodada     | 1ª, 2ª, … 38ª                                  |
| Placar     | Ex.: `2 x 1` ou **`-`** se pendente           |
| Mandante / Visitante | Times                          |

Compartilhe a planilha com o `client_email` das secrets. Use as **mesmas secrets** do velocímetro (`[connections.gsheets]`). No app, clique em **Recarregar planilha** após editar.

**Fallback local:** `dados/calendario_brasileirao_2026.xlsx` (se secrets ausentes).

## Modos de projeção

- **Regressão linear:** `pts ~ rodada + indicador_casa + rodada × indicador_casa`
- **Média simples única:** média de pontos por jogo no intervalo (casa e fora juntos)
- **Média simples separada:** médias distintas em casa e fora
- **Repetir 1º turno:** espelha ida/volta já disputada; fallback sempre por regressão linear

Desempate na classificação: vitórias → saldo de gols → gols marcados → confronto direto entre empatados.

## Deploy Streamlit Cloud

- Main file: `app.py`
- Python 3.10+
- Secrets: copie `[connections.gsheets]` do velocímetro (ver `.streamlit/secrets.toml.example`)

## Estrutura

```
app.py
brasileirao_projecao_core.py   # lógica
brasileirao_gsheets.py         # Google Sheets (secrets velocímetro)
brasileirao_estilo.py          # visual
dados/calendario_brasileirao_2026.xlsx  # fallback local
requirements.txt
```
