import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

df = pd.read_csv('data/raw/Tweets.csv')

print("Dataset Shape:", df.shape)
print("\nColunas:", df.columns.tolist())
print("\nPrimeiras linhas:")
print(df.head())


def analise_sentimento_por_companhia():
    sentiment_airline = df.groupby(['airline', 'airline_sentiment']).size().unstack(fill_value=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sentiment_airline.plot(kind='bar', stacked=True, ax=ax1, color=['#d62728', '#ff7f0e', '#2ca02c'])
    ax1.set_title('Distribuicao de Sentimento por Companhia Aerea')
    ax1.set_xlabel('Companhia Aerea')
    ax1.set_ylabel('Numero de Tweets')
    ax1.legend(title='Sentimento')
    ax1.tick_params(axis='x', rotation=45)
    
    sentiment_pct = sentiment_airline.div(sentiment_airline.sum(axis=1), axis=0) * 100
    sentiment_pct.plot(kind='bar', stacked=True, ax=ax2, color=['#d62728', '#ff7f0e', '#2ca02c'])
    ax2.set_title('Porcentagem de Sentimento por Companhia Aerea')
    ax2.set_xlabel('Companhia Aerea')
    ax2.set_ylabel('Porcentagem')
    ax2.legend(title='Sentimento')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('reports/figures/sentimento_por_companhia.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTotal de tweets por companhia:")
    print(df['airline'].value_counts())
    print("\nDistribuicao geral de sentimentos:")
    print(df['airline_sentiment'].value_counts())
    print("\nPorcentagem de sentimentos negativos por companhia:")
    negative_pct = (sentiment_airline['negative'] / sentiment_airline.sum(axis=1) * 100).sort_values(ascending=False)
    print(negative_pct)


def analise_motivos_negativos():
    negative_df = df[df['airline_sentiment'] == 'negative'].copy()
    
    negative_reasons = negative_df['negativereason'].value_counts().head(10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    negative_reasons.plot(kind='barh', ax=ax1, color='#d62728')
    ax1.set_title('Top 10 Motivos de Feedback Negativo')
    ax1.set_xlabel('Numero de Mencoes')
    ax1.set_ylabel('Motivo')
    ax1.invert_yaxis()
    
    reason_by_airline = pd.crosstab(negative_df['airline'], negative_df['negativereason'])
    top_reasons = negative_reasons.head(5).index
    reason_by_airline[top_reasons].plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Top 5 Motivos Negativos por Companhia')
    ax2.set_xlabel('Companhia Aerea')
    ax2.set_ylabel('Numero de Mencoes')
    ax2.legend(title='Motivo', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('reports/figures/motivos_negativos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTop 10 motivos de feedback negativo:")
    print(negative_reasons)
    print(f"\nTotal de tweets negativos: {len(negative_df)}")
    print(f"Tweets negativos com motivo especificado: {negative_df['negativereason'].notna().sum()}")
    
    print("\nMotivos mais comuns por companhia:")
    for airline in negative_df['airline'].unique():
        airline_negative = negative_df[negative_df['airline'] == airline]
        top_reason = airline_negative['negativereason'].value_counts().head(1)
        if not top_reason.empty:
            print(f"{airline}: {top_reason.index[0]} ({top_reason.values[0]} mencoes)")


print("\n" + "="*60)
print("ANALISE 1: SENTIMENTO POR COMPANHIA AEREA")
print("="*60)
analise_sentimento_por_companhia()

print("\n" + "="*60)
print("ANALISE 2: MOTIVOS DE FEEDBACK NEGATIVO")
print("="*60)
analise_motivos_negativos()
