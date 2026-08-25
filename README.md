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
4. **Probabilístico (placar)** — lê a base semanal em `artifacts/prob_ml/` (sem retreino no app)

Desempate na classificação: vitórias → saldo de gols → gols marcados → confronto direto entre empatados.

## Automação 100% (sem trocar base na mão)

Fluxo desejado:

1. Você só atualiza placares na aba **`Jogos`**
2. Toda **segunda 03:00** o job treina e **publica sozinho** as abas de modelo na **mesma planilha**
3. O Streamlit lê `Jogos` + abas de modelo — sem upload manual

### Setup (uma vez)

1. Baixe o JSON da **service account** (a mesma do Streamlit / `[connections.gsheets]`).
2. Na planilha  
   https://docs.google.com/spreadsheets/d/1QkOIvRa9YinnOveOK4BkX4h_ZtGYRg1ZLzIfokbQh5I/edit  
   compartilhe com o `client_email` da SA como **Editor** (não só leitor).
3. No `.env` do projeto (o agendador já aponta para o repo):

```env
GOOGLE_SERVICE_ACCOUNT_FILE=C:\Users\kaleb\caminho\service-account.json
```

4. (Opcional) Para também sobrescrever um XLSX no Drive:

```env
MODELOS_DRIVE_FILE_ID=id_do_arquivo_no_drive
```

   Compartilhe esse arquivo com a mesma SA como **Editor**.

5. Teste:

```bash
python scripts/weekly_retrain.py --budget fast --skip-download --no-backtest
```

No log deve aparecer `Sheets publicada: … abas`. Depois disso, não precisa mais copiar XLSX para o Drive manualmente.

### Alternativa: pasta do Google Drive for Desktop

Se preferir só arquivo local sincronizado: faça o job gravar `brasileirao_modelos.xlsx` dentro da pasta sincronizada do Drive. O app em Cloud, porém, continua dependendo das **abas na Sheets** (ou de baixar o XLSX) — por isso a publicação na planilha é o caminho recomendado.

O Streamlit **não baixa FPT e não retreina**. Depende só de:

1. **Planilha de resultados** — aba **`Jogos`** (Sheets ou `dados/calendario_brasileirao_2026.xlsx`)
2. **Base de modelos** — arquivo **`brasileirao_modelos.xlsx`** (junto aos resultados) **ou** as mesmas abas coladas na planilha Sheets

Arquivo gerado toda segunda 03:00 em `dados/brasileirao_modelos.xlsx` e `Downloads/brasileirao_modelos.xlsx`.

### Abas do `brasileirao_modelos.xlsx` (não renomear)

| Aba | Conteúdo |
|-----|----------|
| `Leia-me` | Meta (data, champion, fingerprint) |
| `Projecoes_Regressao` | xPts da regressão por jogo |
| `Coefs_Regressao` | Coeficientes / significância |
| `Classif_Regressao` | Classificação projetada (regressão) |
| `Projecoes_Prob` | xPts / λ / 1X2 (probabilístico) |
| `Match_Forecasts` | Previsões detalhadas de placar |
| `Classif_Prob_MC` | Classificação + probs Monte Carlo |
| `Metricas_Prob` | Métricas OOF |
| `Base_Contexto` | Treino com descanso + jogos importantes |
| `Overlay_Calendario` | Relatório do overlay mid-week |

Aba de resultados (você atualiza): **`Jogos`**.

| Modo | Fonte |
|------|--------|
| Média / Repetir 1º turno | Só aba `Jogos` |
| Regressão | `Projecoes_Regressao` (+ gap-fill média se faltar jogo) |
| Probabilístico | `Projecoes_Prob` / `Match_Forecasts` / `Classif_Prob_MC` |

```bash
# Segunda 03:00: gera brasileirao_modelos.xlsx
python scripts/weekly_retrain.py --budget fast
```

### Modelos disponíveis (challengers)

- `league_mean`, `elo_result`, `independent_poisson`, `dixon_coles`
- `poisson_glm`, `elastic_net_goals`
- Ensemble: média / pesos por NLL OOF / blend constrito
- Calibração: none / temperature

Champion, pesos e NLL só são gravados após backtest real (`artifacts/prob_ml/status.json`). Sem histórico multi-temporada FPT, o status pode permanecer `not_evaluated` ou treinar só no calendário 2026.

### Google Drive (base FPT)

File ID configurado: `12rP2nVmKF-VMyY1Oq47Oo7H34ADfxbTV`

1. No Google Drive, compartilhe o arquivo (**Leitor**) com:  
   `streamlit-bot@bot-promocional.iam.gserviceaccount.com`  
   (mesma conta das secrets `[connections.gsheets]` do Streamlit)
2. Secrets/env: `GOOGLE_SERVICE_ACCOUNT_JSON` (ou `GOOGLE_SERVICE_ACCOUNT_FILE`) + opcionalmente `GOOGLE_DRIVE_FILE_ID`
3. Em `config/prob_ml.yaml`: `data.source: google_drive`
4. Localmente, enquanto o share não estiver ativo, o app usa `dados/fpt_matches.csv` (gitignored)

```bash
python scripts/train_pipeline.py --budget fast
```

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
