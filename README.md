# Projeção Brasileirão 2026

App Streamlit para projetar a classificação final do Campeonato Brasileiro com base em resultados reais e modelos configuráveis.

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Atualizar resultados

Edite a planilha:

`dados/calendario_brasileirao_2026.xlsx`

| Coluna     | Descrição                                      |
|------------|------------------------------------------------|
| Rodada     | 1ª, 2ª, … 38ª                                  |
| Data/Hora  | Data e horário do jogo                         |
| Mandante   | Time da casa                                   |
| Placar     | Ex.: `2 x 1` ou `-` se ainda não jogou         |
| Visitante  | Time visitante                                 |
| Estadio    | Opcional                                       |

Salve o arquivo e recarregue o app.

## Modos de projeção

- **Regressão linear:** betas (pts/rodada) no intervalo escolhido; preenche jogos pendentes.
- **Repetir 1º turno:** espelha ida/volta já disputada; fallback por beta.

## Deploy Streamlit Cloud

- Main file: `app.py`
- Python 3.10+

## Estrutura

```
app.py
brasileirao_projecao_core.py   # lógica
brasileirao_estilo.py          # visual
dados/calendario_brasileirao_2026.xlsx
requirements.txt
```
