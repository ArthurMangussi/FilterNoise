# Assessing Adversarial Effects of Noise in Missing Data Imputation

Codebase for the conference paper: *Assessing Adversarial Effects of Noise in Missing Data Imputation*, accepted and presented at the [34th Brazilian Conference on Intelligent Systems (BRACIS)](https://bracis.sbc.org.br/2024/34th-brazilian-conference-on-intelligent-systems-bracis/)

## Paper Details
- Authors: Arthur Dantas Mangussi, Ricardo Cardoso Pereira, Pedro Henriques Abreu, and Ana Carolina Lorena
- Abtract: In real-world scenarios, a wide variety of datasets contain inconsistencies. 
One example of such inconsistency is missing data (MD),
which refers to the absence of information in one or more variables. Missing
imputation strategies emerged as a possible solution for addressing
this problem, which can replace the missing values based on mean, median,
or Machine Learning (ML) techniques. The performance of such
strategies depends on multiple factors. One factor that influences the
missing value imputation (MVI) methods is the presence of noisy instances,
described as anything that obscures the relationship between
the features of an instance and its class, having an adversarial effect.
However, the interaction between MD and noisy instances has received
little attention in the literature. This work fills this gap by investigating
missing and noisy data interplay. Our experimental setup begins
with generating missingness under the Missing Not at Random (MNAR)
mechanism in a multivariate scenario and performing imputation using
seven state-of-the-art MVI methods. Our methodology involves applying
a noise filter before performing the imputation task and evaluating the
quality of the imputation directly. Additionally, we measure the classification
performance with the new estimates. This approach is applied
to both synthetic data and 11 real-world datasets. The effects of noise
filtering before imputation are evaluated. The results show that noise
preprocessing before the imputation task improves the imputation quality
and the classification performance for imputed datasets.
- Year: 2024
- Published in: Will be available as soon as the conference proceedings are published.
- DOI: Will be available as soon as the conference proceedings are published.
- Contact: mangussiarthur@gmail.com

## Paper and Presentation
- The original paper could be acess [here](presentations/BRACIS2024.pdf)
- The PDF presentation is available [here](presentations/Apresentação_BRACIS2024.pdf)

## Dependencies
You'll need a working Python environment to run the code. The required dependencies are specified in the file `requirements.txt`.

You can install all required dependencies by running:
```bash
pip install -r requirements.txt
```

## Citation
If you use this work, please cite:

Bibtex entry:
```bash
Will be available as soon as the conference proceedings are published.
```
## Acknowledgements
The authors gratefully acknowledge the Brazilian funding agencies FAPESP (Fundação Amparo à Pesquisa do Estado de São Paulo) under grants 2022/10553-6, 2023/13688-2, and 2021/06870-3. Moreover, this research was supported by Portuguese Recovery and Resilience Plan (PRR) through project C645008882-00000055 Center for Responsable AI.