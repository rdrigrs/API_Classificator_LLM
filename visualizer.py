import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ResultsVisualizer:
    def __init__(self, results_dir='results', output_dir=None):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir or results_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = None
        self.raw_results = None
        self.per_category_df = None
    
    def load_latest_metrics(self):
        json_files = sorted(self.results_dir.glob('metrics_*.json'), reverse=True)
        if not json_files:
            raise FileNotFoundError(f"Nenhum arquivo de métricas encontrado em {self.results_dir}")
        
        latest_json = json_files[0]
        print(f"Carregando métricas de: {latest_json}")
        
        with open(latest_json, 'r', encoding='utf-8') as f:
            self.metrics = json.load(f)
        
        csv_files = sorted(self.results_dir.glob('metrics_per_category_*.csv'), reverse=True)
        if csv_files:
            self.per_category_df = pd.read_csv(csv_files[0], sep=';')
            print(f"Carregando métricas por categoria de: {csv_files[0]}")
        
        raw_files = sorted(self.results_dir.glob('raw_results_*.csv'), reverse=True)
        if raw_files:
            self.raw_results = pd.read_csv(raw_files[0], sep=';')
            print(f"Carregando resultados brutos de: {raw_files[0]}")
        
        return self.metrics
    
    def load_specific_metrics(self, metrics_file):
        metrics_path = self.results_dir / metrics_file if not Path(metrics_file).is_absolute() else Path(metrics_file)
        
        with open(metrics_path, 'r', encoding='utf-8') as f:
            self.metrics = json.load(f)
        
        timestamp = metrics_path.stem.replace('metrics_', '')
        
        per_category_file = self.results_dir / f'metrics_per_category_{timestamp}.csv'
        if per_category_file.exists():
            self.per_category_df = pd.read_csv(per_category_file, sep=';')
        
        raw_file = self.results_dir / f'raw_results_{timestamp}.csv'
        if raw_file.exists():
            self.raw_results = pd.read_csv(raw_file, sep=';')
        
        return self.metrics
    
    def plot_overall_metrics(self):
        if not self.metrics:
            raise ValueError("Métricas não carregadas. Execute load_latest_metrics() primeiro.")
        
        overall = self.metrics['overall_metrics']
        
        metrics_names = ['F1 Macro', 'F1 Micro', 'Accuracy', 'Precision Macro', 'Recall Macro', 'Krippendorff Alpha']
        metrics_values = [
            overall.get('f1_macro', 0),
            overall.get('f1_micro', 0),
            overall.get('accuracy', 0),
            overall.get('precision_macro', 0),
            overall.get('recall_macro', 0),
            overall.get('krippendorff_alpha', 0)
        ]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(metrics_names, metrics_values, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c'])
        
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Métricas Gerais do Classificador', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        for i, (bar, value) in enumerate(zip(bars, metrics_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.axhline(y=0.80, color='r', linestyle='--', linewidth=2, label='Meta (0.80)')
        ax.legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filename = self.output_dir / 'overall_metrics.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {filename}")
        plt.close()
    
    def plot_per_category_metrics(self):
        if self.per_category_df is None or len(self.per_category_df) == 0:
            print("⚠️  Dados de métricas por categoria não disponíveis.")
            return
        
        df = self.per_category_df.copy()
        df = df.sort_values('f1_score', ascending=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Métricas por Categoria', fontsize=16, fontweight='bold', y=0.995)
        
        x_pos = range(len(df))
        categories = df['category'].tolist()
        
        axes[0, 0].barh(x_pos, df['precision'], color='#3498db', alpha=0.8)
        axes[0, 0].set_yticks(x_pos)
        axes[0, 0].set_yticklabels(categories)
        axes[0, 0].set_xlabel('Precision', fontweight='bold')
        axes[0, 0].set_title('Precision por Categoria', fontweight='bold')
        axes[0, 0].set_xlim(0, 1.0)
        axes[0, 0].grid(axis='x', alpha=0.3)
        for i, v in enumerate(df['precision']):
            axes[0, 0].text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')
        
        axes[0, 1].barh(x_pos, df['recall'], color='#2ecc71', alpha=0.8)
        axes[0, 1].set_yticks(x_pos)
        axes[0, 1].set_yticklabels(categories)
        axes[0, 1].set_xlabel('Recall', fontweight='bold')
        axes[0, 1].set_title('Recall por Categoria', fontweight='bold')
        axes[0, 1].set_xlim(0, 1.0)
        axes[0, 1].grid(axis='x', alpha=0.3)
        for i, v in enumerate(df['recall']):
            axes[0, 1].text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')
        
        axes[1, 0].barh(x_pos, df['f1_score'], color='#9b59b6', alpha=0.8)
        axes[1, 0].set_yticks(x_pos)
        axes[1, 0].set_yticklabels(categories)
        axes[1, 0].set_xlabel('F1-Score', fontweight='bold')
        axes[1, 0].set_title('F1-Score por Categoria', fontweight='bold')
        axes[1, 0].set_xlim(0, 1.0)
        axes[1, 0].grid(axis='x', alpha=0.3)
        for i, v in enumerate(df['f1_score']):
            axes[1, 0].text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')
        
        axes[1, 1].barh(x_pos, df['support'], color='#e74c3c', alpha=0.8)
        axes[1, 1].set_yticks(x_pos)
        axes[1, 1].set_yticklabels(categories)
        axes[1, 1].set_xlabel('Support (Número de Amostras)', fontweight='bold')
        axes[1, 1].set_title('Número de Amostras por Categoria', fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
        for i, v in enumerate(df['support']):
            axes[1, 1].text(v + 1, i, f'{int(v)}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        filename = self.output_dir / 'per_category_metrics.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {filename}")
        plt.close()
    
    def plot_metrics_comparison(self):
        if self.per_category_df is None or len(self.per_category_df) == 0:
            print("⚠️  Dados de métricas por categoria não disponíveis.")
            return
        
        df = self.per_category_df.copy()
        df = df.sort_values('f1_score', ascending=False)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = range(len(df))
        width = 0.25
        
        ax.bar([i - width for i in x], df['precision'], width, label='Precision', color='#3498db', alpha=0.8)
        ax.bar(x, df['recall'], width, label='Recall', color='#2ecc71', alpha=0.8)
        ax.bar([i + width for i in x], df['f1_score'], width, label='F1-Score', color='#9b59b6', alpha=0.8)
        
        ax.set_xlabel('Categorias', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Comparação de Métricas por Categoria', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(df['category'], rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        filename = self.output_dir / 'metrics_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {filename}")
        plt.close()
    
    def generate_all_visualizations(self):
        print("🎨 Gerando visualizações dos resultados...\n")
        
        try:
            self.plot_overall_metrics()
            print("✅ Métricas gerais geradas\n")
        except Exception as e:
            print(f"❌ Erro ao gerar métricas gerais: {e}\n")
        
        try:
            self.plot_per_category_metrics()
            print("✅ Métricas por categoria geradas\n")
        except Exception as e:
            print(f"❌ Erro ao gerar métricas por categoria: {e}\n")
        
        try:
            self.plot_metrics_comparison()
            print("✅ Comparação de métricas gerada\n")
        except Exception as e:
            print(f"❌ Erro ao gerar comparação: {e}\n")
        
        print("✨ Visualizações concluídas!")


def main():
    import sys
    
    visualizer = ResultsVisualizer()
    
    if len(sys.argv) > 1:
        metrics_file = sys.argv[1]
        visualizer.load_specific_metrics(metrics_file)
    else:
        visualizer.load_latest_metrics()
    
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()

