# API Classificator LLM

Classificador de APIs FinTech usando Google Gemini LLM.

## Configuração

### 1. Ambiente Virtual

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Configurar Chave de API

**⚠️ IMPORTANTE: Nunca commite sua chave de API no código!**

#### Opção 1: Arquivo .env (Recomendado)

1. Copie o arquivo de exemplo:
```bash
cp env.example .env
```

2. Edite o arquivo `.env` e adicione sua chave:
```
GEMINI_API_KEY=sua_chave_api_aqui
```

#### Opção 2: Variável de Ambiente

```bash
export GEMINI_API_KEY="sua_chave_api_aqui"
```

Para obter sua chave de API do Google Gemini:
- Acesse: https://aistudio.google.com/app/apikey
- Crie uma nova chave de API
- Copie e cole no arquivo `.env` ou configure como variável de ambiente

### 3. Executar o Classificador

```bash
# Com o venv ativado
python classificator.py

# Ou usando o Python do venv diretamente
./venv/bin/python classificator.py
```

#### Limitar número de APIs (para testes)

Para testar rapidamente sem processar todo o dataset, configure `MAX_APIS` no arquivo `.env`:

```bash
# Processar apenas as primeiras 10 APIs
MAX_APIS=10
```

Se `MAX_APIS=0` ou não estiver definido, todas as APIs do dataset serão processadas.

## Estrutura de Saída

Os resultados são salvos no diretório `results/`:

- `raw_results_YYYYMMDD_HHMMSS.csv` - Resultados brutos das classificações
- `metrics_YYYYMMDD_HHMMSS.json` - Métricas completas em JSON
- `metrics_summary_YYYYMMDD_HHMMSS.csv` - Resumo das métricas principais
- `metrics_per_category_YYYYMMDD_HHMMSS.csv` - Métricas detalhadas por categoria

## Variáveis de Ambiente

Todas as configurações podem ser definidas via variáveis de ambiente ou arquivo `.env`:

- `GEMINI_API_KEY` (obrigatório) - Chave de API do Google Gemini
- `DATASET_PATH` (opcional) - Caminho do dataset CSV (padrão: `fintechapis.csv`)
- `NUM_RUNS` (opcional) - Número de execuções para confiabilidade (padrão: `5`)
- `GEMINI_MODEL` (opcional) - Modelo do Gemini (padrão: `gemini-2.5-flash`)
- `TEMPERATURE` (opcional) - Temperatura para geração (padrão: `0.0`)
- `OUTPUT_DIR` (opcional) - Diretório de saída (padrão: `results`)
- `MAX_APIS` (opcional) - Número máximo de APIs a classificar (padrão: `0` = todas). Útil para testes rápidos.

## Visualização de Resultados

Após executar o classificador, você pode gerar gráficos e visualizações dos resultados usando o script `visualizer.py`:

```bash
# Gerar visualizações dos resultados mais recentes
python visualizer.py

# Ou especificar um arquivo de métricas específico
python visualizer.py results/metrics_20251107_204147.json
```

### Gráficos Gerados

O visualizador gera os seguintes gráficos no diretório `results/`:

1. **`overall_metrics.png`** - Métricas gerais (F1 Macro/Micro, Accuracy, Precision, Recall, Krippendorff Alpha)
2. **`per_category_metrics.png`** - Métricas detalhadas por categoria (Precision, Recall, F1-Score, Support)
3. **`metrics_comparison.png`** - Comparação lado a lado de Precision, Recall e F1-Score por categoria

Todos os gráficos são salvos em alta resolução (300 DPI) e prontos para apresentações ou relatórios.

## Segurança

- O arquivo `.env` está no `.gitignore` e não será commitado
- Nunca compartilhe sua chave de API
- Use variáveis de ambiente em ambientes de produção


## Recursos

[Dataset](https://github.com/UQAR-TUW/FinTechAPIsDataset)