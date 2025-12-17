# API Classificator LLM

Classificador de APIs FinTech usando LLMs (Google Gemini ou DeepSeek).

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

2. Edite o arquivo `.env` e configure o provedor e a chave:

**Para usar Google Gemini:**
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=sua_chave_api_aqui
```

**Para usar DeepSeek:**
```
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sua_chave_api_aqui
```

**Para usar Groq:**
```
LLM_PROVIDER=groq
GROQ_API_KEY=sua_chave_api_aqui
```

#### Opção 2: Variável de Ambiente

```bash
# Para Gemini
export LLM_PROVIDER="gemini"
export GEMINI_API_KEY="sua_chave_api_aqui"

# Para DeepSeek
export LLM_PROVIDER="deepseek"
export DEEPSEEK_API_KEY="sua_chave_api_aqui"

# Para Groq
export LLM_PROVIDER="groq"
export GROQ_API_KEY="sua_chave_api_aqui"
```

#### Obter Chaves de API

**Google Gemini:**
- Acesse: https://aistudio.google.com/app/apikey
- Crie uma nova chave de API

**DeepSeek:**
- Acesse: https://platform.deepseek.com
- Crie uma nova chave de API

**Groq:**
- Acesse: https://console.groq.com
- Crie uma nova chave de API

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

| Variável | Obrigatório | Padrão | Descrição |
|----------|-------------|--------|-----------|
| `LLM_PROVIDER` | Não | `gemini` | Provedor de LLM (`gemini`, `deepseek` ou `groq`) |
| `GEMINI_API_KEY` | Se provider=gemini | - | Chave de API do Google Gemini |
| `DEEPSEEK_API_KEY` | Se provider=deepseek | - | Chave de API do DeepSeek |
| `GROQ_API_KEY` | Se provider=groq | - | Chave de API do Groq |
| `GEMINI_MODEL` | Não | `gemini-2.5-flash` | Modelo do Gemini |
| `DEEPSEEK_MODEL` | Não | `deepseek-chat` | Modelo do DeepSeek (`deepseek-chat`, `deepseek-reasoner`) |
| `GROQ_MODEL` | Não | `llama-3.3-70b-versatile` | Modelo do Groq (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`) |
| `DATASET_PATH` | Não | `fintechapis.csv` | Caminho do dataset CSV |
| `NUM_RUNS` | Não | `5` | Número de execuções para confiabilidade |
| `TEMPERATURE` | Não | `0.0` | Temperatura para geração |
| `OUTPUT_DIR` | Não | `results` | Diretório de saída |
| `MAX_APIS` | Não | `0` (todas) | Número máximo de APIs a classificar |

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