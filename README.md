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

Todos os modos projetam **pontuação decimal por jogo** (ex.: 2,2 pts), sem arredondar para 3/1/0.
Nas regressões, o ganho por rodada é limitado a **no máximo 3 pontos**.

1. **Regressão de Momento e Aceleração (efeitos fixos)** — Pontos Acumulados ~ Efeito Fixo do Time + Rodada + Rodada ao Quadrado + Interação Rodada × Time + Interação Rodada ao Quadrado × Time + Forma Recente
2. **Regressão de Momento e Histórico (efeitos fixos)** — Pontos Acumulados ~ Efeito Fixo do Time + Rodada + Interação Rodada × Time + Proporção Casa + Força dos Adversários Passados
3. **Regressão Completa (efeitos fixos)** — Pontos Acumulados ~ Efeito Fixo do Time + Rodada + Rodada ao Quadrado + Interação Rodada × Time + Interação Rodada ao Quadrado × Time + Forma Recente + Força dos Adversários Passados + Proporção Casa
4. **Média casa x fora × forma recente**
5. **Média casa x fora** — só média pts/jogo em casa ou fora (sem forma recente)
6. **Repetir 1º turno** — espelha ida/volta já disputada; fallback pela média × forma recente
7. **Regressão Completa com Limites (efeitos fixos)** — como a Completa, com Rodada Centrada ao Quadrado (Rodada − 19) e forma recente com peso decrescente (80% → 50% em 5 rodadas, piso 25%)

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
