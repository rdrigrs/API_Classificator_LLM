import os
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import krippendorff
import time

load_dotenv()


class Config:
    DATASET_PATH = os.getenv('DATASET_PATH', 'fintechapis.csv')
    API_KEY = os.getenv('GEMINI_API_KEY', '')
    NUM_RUNS = int(os.getenv('NUM_RUNS', '5'))
    MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.0'))
    OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', 'results'))
    _max_apis_env = os.getenv('MAX_APIS', '0')
    MAX_APIS = int(_max_apis_env) if _max_apis_env and _max_apis_env != '0' else None
    
    CATEGORY_DEFINITIONS = """
<categories>
- banking: Serviços bancários tradicionais, incluindo gestão de contas, caixas automáticos (ATMs), gestão de cartões de crédito e métodos de pagamento.
- blockchain: Tecnologia Blockchain, como criptomoedas e contratos inteligentes.
- client: Gestão de informações do cliente, incluindo perfis de clientes, metas pessoais e classificação de crédito.
- currency: Taxas de câmbio e ferramentas de conversão de moeda.
- payment: Processamento de pagamentos, incluindo transações, carteiras digitais e faturas.
- savings: Ferramentas de planejamento financeiro para poupanças, planos de investimento, cálculos de juros e produtos de poupança.
- trading: Atividades de negociação (trading), negociação de ações, forex e carteiras de investimento.
- transfer: Transferência de fundos entre contas, tanto domésticas quanto internacionais.
- user-password: Autenticação de usuário, gestão de senhas e protocolos de segurança (tokens) para acesso do usuário.
- loan-mortgage: Processos de empréstimos e hipotecas, submissão de aplicações e credores.
</categories>
"""
    
    INSTRUCTIONS_TEMPLATE = """
{category_definitions}

<instructions>
1. Você receberá um resumo de API (API summary) do setor financeiro. Leia-o atentamente.
2. Identifique a funcionalidade principal e o propósito desta API.
3. Classifique a API em uma das seguintes 10 categorias: [banking, blockchain, client, currency, payment, savings, trading, transfer, user-password, loan-mortgage].
4. Escreva seu processo de raciocínio em duas sentenças, estritamente dentro das tags <thinking>.
5. Responda com o nome da categoria ESTREITAMENTE dentro das tags <category>.
</instructions>

user: {api_content}
"""


class CategoryExtractor:
    @staticmethod
    def extract(response_text):
        match = re.search(r'<category>\s*([^<]+?)\s*</category>', response_text, re.IGNORECASE)
        if match:
            return match.group(1).lower().strip()
        return "error: no category found"


class APIClassifier:
    FALLBACK_MODELS = [
        'gemini-2.5-flash',
        'gemini-2.5-pro',
        'gemini-2.0-flash-exp',
        'gemini-1.5-pro',
        'gemini-pro'
    ]
    
    def __init__(self, api_key, model=None, temperature=None):
        self.client = genai.Client(api_key=api_key)
        self.model = model or Config.MODEL
        self.temperature = temperature or Config.TEMPERATURE
        self.extractor = CategoryExtractor()
        self.validated_model = None
    
    def _try_model(self, model_name, prompt):
        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"temperature": self.temperature}
            )
            return response.text, True
        except Exception as e:
            return str(e), False
    
    def classify(self, api_content):
        prompt = Config.INSTRUCTIONS_TEMPLATE.format(
            category_definitions=Config.CATEGORY_DEFINITIONS,
            api_content=api_content
        )
        
        models_to_try = [self.model] + [m for m in self.FALLBACK_MODELS if m != self.model]
        
        for model_name in models_to_try:
            response_text, success = self._try_model(model_name, prompt)
            if success:
                if self.validated_model != model_name:
                    print(f"Usando modelo: {model_name}")
                    self.validated_model = model_name
                return self.extractor.extract(response_text)
            else:
                if "404" in str(response_text) or "NOT_FOUND" in str(response_text):
                    continue
            time.sleep(10)
        print(f"Erro: Nenhum modelo disponível. Último erro: {response_text}")
        return "error: api call failed"


class MetricsCalculator:
    def __init__(self, results_df, classification_cols):
        self.results_df = results_df
        self.classification_cols = classification_cols
        self.metrics = {}
    
    def calculate_consensus_prediction(self):
        self.results_df['model_prediction'] = (
            self.results_df[self.classification_cols]
            .mode(axis=1)[0]
            .fillna('error: indeterminate mode')
        )
        return self.results_df
    
    def calculate_f1_metrics(self):
        valid_results = self.results_df[
            ~self.results_df['model_prediction'].str.contains('error', case=False)
        ]
        
        if len(valid_results) == 0:
            return {
                'f1_macro': 0.0,
                'f1_micro': 0.0,
                'accuracy': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                'valid_samples': 0,
                'total_samples': len(self.results_df)
            }
        
        y_true = valid_results['consensus']
        y_pred = valid_results['model_prediction']
        
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        return {
            'f1_macro': float(f1_macro),
            'f1_micro': float(f1_micro),
            'accuracy': float(accuracy),
            'precision_macro': float(precision),
            'recall_macro': float(recall),
            'valid_samples': len(valid_results),
            'total_samples': len(self.results_df)
        }
    
    def calculate_krippendorff_alpha(self):
        data_for_ka = self.results_df[self.classification_cols].values
        
        unique_labels = sorted(list(set(data_for_ka.flatten())))
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        int_data = np.array([
            [label_to_int.get(label, -1) for label in row] 
            for row in data_for_ka
        ])
        
        try:
            ka_result = krippendorff.alpha(
                int_data.T,
                level_of_measurement='nominal',
                value_domain=list(label_to_int.values())
            )
            return {
                'krippendorff_alpha': float(ka_result) if not np.isnan(ka_result) else 0.0,
                'unique_labels_count': len(unique_labels)
            }
        except Exception as e:
            print(f"Erro ao calcular Krippendorff's Alpha: {e}")
            return {
                'krippendorff_alpha': 0.0,
                'unique_labels_count': len(unique_labels)
            }
    
    def calculate_per_category_metrics(self):
        valid_results = self.results_df[
            ~self.results_df['model_prediction'].str.contains('error', case=False)
        ]
        
        if len(valid_results) == 0:
            return {}
        
        y_true = valid_results['consensus']
        y_pred = valid_results['model_prediction']
        
        categories = sorted(set(y_true) | set(y_pred))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=categories, zero_division=0
        )
        
        per_category = {}
        for i, cat in enumerate(categories):
            per_category[cat] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        return per_category
    
    def calculate_all_metrics(self):
        self.calculate_consensus_prediction()
        
        f1_metrics = self.calculate_f1_metrics()
        ka_metrics = self.calculate_krippendorff_alpha()
        per_category = self.calculate_per_category_metrics()
        
        self.metrics = {
            'timestamp': datetime.now().isoformat(),
            'experiment_config': {
                'num_runs': len(self.classification_cols),
                'total_apis': len(self.results_df),
                'model': Config.MODEL,
                'temperature': Config.TEMPERATURE
            },
            'overall_metrics': {
                **f1_metrics,
                **ka_metrics
            },
            'per_category_metrics': per_category
        }
        
        return self.metrics


class MetricsExporter:
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir or Config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_json(self, metrics, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Métricas JSON salvas em: {filepath}")
        return filepath
    
    def save_csv_summary(self, metrics, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_summary_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        summary_data = {
            'metric': [],
            'value': []
        }
        
        overall = metrics['overall_metrics']
        summary_data['metric'].extend([
            'f1_macro', 'f1_micro', 'accuracy', 'precision_macro',
            'recall_macro', 'krippendorff_alpha', 'valid_samples', 'total_samples'
        ])
        summary_data['value'].extend([
            overall.get('f1_macro', 0),
            overall.get('f1_micro', 0),
            overall.get('accuracy', 0),
            overall.get('precision_macro', 0),
            overall.get('recall_macro', 0),
            overall.get('krippendorff_alpha', 0),
            overall.get('valid_samples', 0),
            overall.get('total_samples', 0)
        ])
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(filepath, index=False, sep=';')
        
        print(f"Resumo de métricas CSV salvo em: {filepath}")
        return filepath
    
    def save_per_category_csv(self, metrics, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_per_category_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        per_category = metrics.get('per_category_metrics', {})
        if not per_category:
            print("Nenhuma métrica por categoria disponível.")
            return None
        
        rows = []
        for category, cat_metrics in per_category.items():
            rows.append({
                'category': category,
                'precision': cat_metrics['precision'],
                'recall': cat_metrics['recall'],
                'f1_score': cat_metrics['f1_score'],
                'support': cat_metrics['support']
            })
        
        df_category = pd.DataFrame(rows)
        df_category.to_csv(filepath, index=False, sep=';')
        
        print(f"Métricas por categoria CSV salvas em: {filepath}")
        return filepath
    
    def save_all(self, metrics):
        self.save_json(metrics)
        self.save_csv_summary(metrics)
        self.save_per_category_csv(metrics)


class ExperimentRunner:
    def __init__(self, config=None):
        self.config = config or Config
        self.classifier = None
        self.results_df = None
        self.metrics = None
    
    def initialize_classifier(self):
        if not self.config.API_KEY:
            raise ValueError(
                "Erro: A chave API do Gemini não foi configurada.\n"
                "Configure a variável de ambiente GEMINI_API_KEY ou crie um arquivo .env com:\n"
                "GEMINI_API_KEY=sua_chave_aqui"
            )
        try:
            self.classifier = APIClassifier(api_key=self.config.API_KEY)
        except Exception as e:
            print(f"Erro: Falha ao inicializar o cliente Gemini: {e}")
            raise
    
    def load_dataset(self):
        try:
            df = pd.read_csv(self.config.DATASET_PATH, sep=';')
            df['consensus'] = df['consensus'].str.lower().str.strip()
            return df
        except FileNotFoundError:
            print(f"Erro: Arquivo não encontrado em {self.config.DATASET_PATH}")
            raise
        except Exception as e:
            print(f"Erro ao carregar ou processar o CSV: {e}")
            raise
    
    def run_classifications(self, df):
        if self.config.MAX_APIS and self.config.MAX_APIS > 0:
            df = df.head(self.config.MAX_APIS)
            print(f"⚠️  Limite de {self.config.MAX_APIS} APIs aplicado. Processando apenas as primeiras {len(df)} APIs.")
        
        classification_cols = [f'run_{i+1}' for i in range(self.config.NUM_RUNS)]
        results_df = df[['filename', 'content', 'consensus']].copy()
        all_runs_classifications = []
        
        total_apis = len(df)
        print(f"Iniciando a classificação para {total_apis} APIs, {self.config.NUM_RUNS} vezes cada...")
        
        for index, row in df.iterrows():
            api_content = row['content']
            current_classifications = []
            
            for run_id in range(self.config.NUM_RUNS):
                classification = self.classifier.classify(api_content)
                current_classifications.append(classification)
                print(f"API {index+1}/{total_apis} - Run {run_id+1}: Classificado como '{classification}'")
            
            all_runs_classifications.append(current_classifications)
        
        results_matrix = np.array(all_runs_classifications)
        for i, col in enumerate(classification_cols):
            results_df[col] = results_matrix[:, i]
        
        self.results_df = results_df
        return results_df, classification_cols
    
    def save_raw_results(self, filename=None):
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"raw_results_{timestamp}.csv"
        
        filepath = self.config.OUTPUT_DIR / filename
        self.results_df.to_csv(filepath, index=False, sep=';')
        print(f"\n--- Resultados Brutos Salvos ---")
        print(f"Arquivo: {filepath}")
        return filepath
    
    def calculate_metrics(self, classification_cols):
        calculator = MetricsCalculator(self.results_df, classification_cols)
        self.metrics = calculator.calculate_all_metrics()
        return self.metrics
    
    def print_metrics_summary(self):
        if not self.metrics:
            return
        
        overall = self.metrics['overall_metrics']
        
        print("\n--- Métricas Finais ---")
        print(f"Macro F1-Score (Acurácia): {overall['f1_macro']:.3f} (Meta: >= 0.80)")
        print(f"Micro F1-Score: {overall['f1_micro']:.3f}")
        print(f"Acurácia: {overall['accuracy']:.3f}")
        print(f"Precisão Macro: {overall['precision_macro']:.3f}")
        print(f"Recall Macro: {overall['recall_macro']:.3f}")
        print(f"Krippendorff's Alpha (Consistência): {overall['krippendorff_alpha']:.3f} (Meta: >= 0.80)")
        print(f"Amostras válidas: {overall['valid_samples']}/{overall['total_samples']}")
        
        print("\n--- Conclusão da PoC ---")
        f1_macro = overall['f1_macro']
        ka_result = overall['krippendorff_alpha']
        
        if f1_macro >= 0.80 and ka_result >= 0.80:
            print("✅ SUCESSO! O LLM demonstrou alto desempenho e consistência.")
        elif f1_macro >= 0.80:
            print("⚠️ Desempenho alto, mas a consistência (k-a) precisa ser melhorada.")
        elif ka_result >= 0.80:
            print("⚠️ Alta consistência, mas o F1-Score precisa ser melhorado.")
        else:
            print("❌ FALHA. O LLM não atingiu as metas de desempenho e consistência.")
    
    def run(self):
        self.initialize_classifier()
        df = self.load_dataset()
        
        results_df, classification_cols = self.run_classifications(df)
        self.save_raw_results()
        
        metrics = self.calculate_metrics(classification_cols)
        self.print_metrics_summary()
        
        exporter = MetricsExporter()
        exporter.save_all(metrics)
        
        return results_df, metrics


def run_experiment():
    runner = ExperimentRunner()
    return runner.run()


if __name__ == "__main__":
    run_experiment()
