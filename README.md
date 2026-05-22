Modelagem termodinâmica em Python para predição de precipitação de parafinas no Pré-Sal (Parceria PRH-ANP / LRAP-UFRJ).

Python 3.8+ | NumPy | SciPy | Matplotlib | MIT License

Este repositório contém modelos computacionais desenvolvidos para cálculos de equilíbrio sólido-líquido e análise termodinâmica da precipitação de parafinas em sistemas petrolíferos. 
O projeto foi desenvolvido no âmbito do programa de pesquisa PRH-ANP focado em aplicações de petróleo, gás e energia.

Sumário

- Visão geral
- Problema físico
- Metodologia
- Estrutura do repositório
- Como executar
- Dependências
- Resultados Gerados
- Referências

Visão geral

Este repositório implementa modelos termodinâmicos para equilíbrio sólido-líquido (SLE) de misturas parafínicas, aplicados à cristalização de parafinas, um problema central em Flow Assurance na indústria de petróleo. A fração parafínica é tratada como uma distribuição contínua (Gamma), discretizada em pseudocomponentes, e o flash sólido-líquido é resolvido com diferentes níveis de sofisticação (coeficientes de atividade, equação de estado Peng-Robinson, modelos Wilson/UNIQUAC e Flory-FV). Um dos scripts ainda realiza otimização paramétrica para ajustar a distribuição a dados experimentais da literatura.

Problema físico

A precipitação de parafinas ocorre quando a temperatura do óleo cai abaixo do ponto de nuvem, formando cristais que podem obstruir dutos e equipamentos. Para prever esse fenômeno, é necessário:
- representar a distribuição de tamanhos moleculares da fração pesada;
- calcular a partição líquido-sólido em função da temperatura;
- considerar não idealidades termodinâmicas.

Metodologia

1. Distribuição Gamma para modelar a fração parafínica (C10-C36 e extensões).
2. Discretização em pseudocomponentes (granularidade ajustável).
3. Cálculo de propriedades (temperatura de fusão, entalpia, volume molar, parâmetros de solubilidade, etc.) via correlações da literatura.
4. Flash sólido-líquido usando método de Rachford-Rice, com:
   - coeficientes de atividade: Wilson, UNIQUAC, Flory-FV;
   - equação de estado Peng-Robinson para a fase líquida.
5. Otimização não linear para ajustar os parâmetros da distribuição Gamma a dados experimentais de cristalização.

Estrutura do repositório

Arquivo: src/01_discretization_demo.py
Descrição: Implementa a discretização da distribuição Gamma de massas molares em diferentes granularidades (1 em 1, 5 em 5, 10 em 10 carbonos).

Arquivo: src/02_flash_sl.py
Descrição: Implementa o flash sólido-líquido simplificado usando método de Rachford‑Rice e distribuição Gamma fixa.

Arquivo: src/03_eos_approach.py
Descrição: Implementa a equação de estado de Peng‑Robinson para a fase líquida, com cálculo de coeficientes de fugacidade e parâmetros de interação. 

Arquivo: src/04_pseudocomp_analysis.py
Descrição: Implementa o ajuste da distribuição Gamma aos dados experimentais de fração molar de n‑alcanos (C18–C36).

Arquivo: src/05_flash_wilson_and_uniquac.py
Descrição: Implementa os modelos de coeficiente de atividade Wilson e UNIQUAC para a fase sólida, juntamente com Flory‑Free Volume para a fase líquida, com validação contra dados experimentais da literatura

Arquivo: src/06_fit_gamma.py
Descrição: Implementa a otimização não linear dos parâmetros da distribuição Gamma (α, β, γ) a partir de uma curva experimental de cristalização.

Como executar

1. Clone o repositório:
   git clone https://github.com/rogeriomaximino/thermodynamic-modeling-of-wax-precipitation

2. Instale as dependências:
   pip install -r requirements.txt

3. Execute um script, por exemplo:
   python src/01_discretization_demo.py

Dependências

- Python 3.8 ou superior
- NumPy
- SciPy
- Matplotlib

Todas listadas no arquivo requirements.txt.

Resultados gerados

A execução de cada script resulta na exibição imediata de gráficos. A seguir, uma síntese do output visual de cada implementação.

01_discretization_demo.py
Gera gráficos comparativos da probabilidade cumulativa entre o modelo contínuo e as versões discretizadas.

02_flash_sl.py
Gera curvas de fração molar da fase sólida por pseudocomponente e a fração mássica total de sólidos em função da temperatura.

03_eos_approach.py
Gera as mesmas curvas de fração sólida vs. temperatura, mas com modelagem mais rigorosa, adequada para sistemas com alta pressão e presença de gases leves.

04_pseudocomp_analysis.py
Gera gráficos comparativos entre a composição original e a modelada, além da definição de pseudocomponentes.

05_flash_wilson_and_uniquac.py
Gera curvas de percentual cristalizado vs. temperatura, comparadas com dados experimentais da literatura, e calcula o erro médio absoluto (MAE) de cada modelo.

06_fit_gamma.py
Gera dois gráficos principais: 
(a) ajuste da porcentagem cristalizada vs. temperatura (modelo vs. artigo); 
(b) fração mássica de cada n‑parafina (modelo vs. artigo).

Referências

- Dauphin et al. (1999)
- Broadhurst (1962)
- Morgan & Kobayashi (1994)

Licença

MIT License.
