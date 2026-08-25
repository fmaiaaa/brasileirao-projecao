# Projeção Brasileirão 2026

App Streamlit para projetar a classificação final do Campeonato Brasileiro com base em resultados reais e modelos configuráveis.

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Testes:

```bash
pytest -q tests
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

Todos os modos (exceto o probabilístico puro de placar) projetam **pontuação decimal por jogo** quando aplicável.
Nas regressões, o ganho por rodada é limitado a **no máximo 3 pontos**.
Quando o modo usa Forma Recente, o peso dela na projeção cai de **80%** (próxima rodada) para **50%** (daqui a 5) até o piso de **20%**, misturando com a forma geral.

1. **Regressão** — efeitos fixos + rodada + forma + adversários + casa
2. **Média casa x fora × forma recente** — com decaimento de peso da forma
3. **Repetir 1º turno** — espelha ida/volta; fallback pela média × forma
4. **Probabilístico (placar)** — novo pipeline `prob_ml` (só treina **quando este modo está selecionado**)

Desempate na classificação: vitórias → saldo de gols → gols marcados → confronto direto entre empatados.

## Pipeline probabilístico (`prob_ml`)

Arquitetura modular para previsão de **matriz de placar** \(P(G_H, G_A)\), ensemble temporal, calibração e Monte Carlo do campeonato.

```text
Base (local | Google Drive)
  → schema map + validação + fingerprint
  → features leakage-safe (shift→rolling/EWMA)
  → ratings (Elo)
  → model zoo + HPO leve + OOF temporal
  → ensemble + calibração
  → score matrix → xPts / 1X2 / O/U / BTTS
  → Monte Carlo (amostra placares)
  → site (modo Probabilístico + Model Lab)
```

### Treino offline (não roda a cada request)

```bash
python scripts/train_pipeline.py --synthetic --budget fast
# com base local:
# configure config/prob_ml.yaml → data.local_path
python scripts/train_pipeline.py --budget standard
```

No Streamlit, o treino/avaliação só dispara ao selecionar o modo **Probabilístico**.

### Modelos disponíveis (challengers)

- `league_mean`, `elo_result`, `independent_poisson`, `dixon_coles`
- `poisson_glm`, `elastic_net_goals`
- Ensemble: média / pesos por NLL OOF / blend constrito
- Calibração: none / temperature

Champion, pesos e NLL só são gravados após backtest real (`artifacts/prob_ml/status.json`). Sem histórico multi-temporada FPT, o status pode permanecer `not_evaluated` ou treinar só no calendário 2026.

### Google Drive (preparado — aguardando link)

1. Compartilhe **um único arquivo** (CSV/Excel/Parquet) com a service account  
2. Configure secrets/env:

- `GOOGLE_DRIVE_FILE_ID` ou `GOOGLE_DRIVE_FILE_URL`
- `GOOGLE_SERVICE_ACCOUNT_JSON` (JSON completo; **nunca** commitar)

3. Em `config/prob_ml.yaml`: `data.source: google_drive`  
4. Ajuste `config/schema_map.yaml` se os nomes de colunas diferirem  
5. Rode `scripts/train_pipeline.py` e selecione o modo no app  

Ver também `.streamlit/secrets.toml.example`.

### Estrutura relevante

```
config/prob_ml.yaml
config/schema_map.yaml
prob_ml/          # data, features, ratings, models, backtesting,
                  # selection, optimization, ensemble, calibration,
                  # simulation, monitoring, pipeline, integration
scripts/train_pipeline.py
tests/test_prob_ml.py
artifacts/prob_ml/
```

## Deploy Streamlit Cloud

- Main file: `app.py`
- Python 3.10+
- Secrets: copie `[connections.gsheets]` do velocímetro (ver `.streamlit/secrets.toml.example`)
- Opcional: secrets Drive para a base FPT

## Estrutura legada

```
app.py
brasileirao_projecao_core.py   # lógica dos modos clássicos
brasileirao_gsheets.py
brasileirao_estilo.py
dados/calendario_brasileirao_2026.xlsx
requirements.txt
```
