import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from pathlib import Path
import numpy as np
import math
from typing import Iterable
from itertools import combinations
import time
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset
notebook_dir = Path().resolve()
movilens = pd.read_csv( notebook_dir / 'movilens_dataset/movies.csv')

# Transform the dataset
movilens['genres'] = movilens['genres'].str.split('|')

# Remove the '(no genres listed)' genre
movilens = movilens[movilens['genres'].apply(lambda x: '(no genres listed)' not in x)]
te = TransactionEncoder()
te_ary = te.fit(movilens['genres']).transform(movilens['genres'])
movilens = pd.DataFrame(te_ary, columns=te.columns_).set_index(movilens['title'])

# -------------------------------------

# Supports
support_Ro_Dr = np.logical_and(movilens['Romance'], movilens['Drama']).mean()
support_Ac_Ad_Th = np.logical_and(np.logical_and(movilens['Action'], movilens['Adventure']), movilens['Thriller']).mean()
support_Cr_Ac_Th = np.logical_and(np.logical_and(movilens['Crime'], movilens['Action']), movilens['Thriller']).mean()
support_Cr_Ch = np.logical_and(movilens['Crime'], movilens['Children']).mean()
# Confidences
confidence_Ro_Dr = support_Ro_Dr / movilens['Romance'].mean()
confidence_AcAd_Th = support_Ac_Ad_Th / np.logical_and(movilens['Action'], movilens['Adventure']).mean()
confidence_CrAc_Th = support_Cr_Ac_Th / np.logical_and(movilens['Crime'], movilens['Action']).mean()
confidence_Cr_AcTh = support_Cr_Ac_Th / movilens['Crime'].mean()
confidence_Cr_Ch = support_Cr_Ch / movilens['Crime'].mean()
# Lifts
lift_Ro_Dr = confidence_Ro_Dr / movilens['Drama'].mean()
lift_AcAd_Th = confidence_AcAd_Th / movilens['Thriller'].mean()
lift_CrAc_Th = confidence_CrAc_Th / movilens['Thriller'].mean()
lift_Cr_AcTh = confidence_Cr_AcTh / np.logical_and(movilens['Action'], movilens['Thriller']).mean()
lift_Cr_Ch = confidence_Cr_Ch / movilens['Children'].mean()

print("Romance -> Drama")
print(f"Suppport (Romance -> Drama): {support_Ro_Dr}")
print(f"Confidence (Romance -> Drama): {confidence_Ro_Dr}")
print(f"Lift (Romance -> Drama): {lift_Ro_Dr}")

print("\nAction, Adventure -> Thriller")
print(f"Suppport (Action, Adventure -> Thriller): {support_Ac_Ad_Th}")
print(f"Confidence (Action, Adventure -> Thriller): {confidence_AcAd_Th}")
print(f"Lift (Action, Adventure -> Thriller): {lift_AcAd_Th}")

print("\nCrime, Action -> Thriller")
print(f"Suppport (Crime, Action -> Thriller): {support_Cr_Ac_Th}")
print(f"Confidence (Crime, Action -> Thriller): {confidence_CrAc_Th}")
print(f"Lift (Crime, Action -> Thriller): {lift_CrAc_Th}")

print("\nCrime -> Action, Thriller")
print(f"Suppport (Crime -> Action, Thriller): {support_Cr_Ac_Th}")
print(f"Confidence (Crime -> Action, Thriller): {confidence_Cr_AcTh}")
print(f"Lift (Crime -> Action, Thriller): {lift_Cr_AcTh}")

print("\nCrime -> Children")
print(f"Suppport (Crime -> Children): {support_Cr_Ch}")
print(f"Confidence (Crime -> Children): {confidence_Cr_Ch}")
print(f"Lift (Crime -> Children): {lift_Cr_Ch}")

# RESULTADOS
# Romance -> Drama
# Suppport (Romance -> Drama): 0.067854084603528
# Confidence (Romance -> Drama): 0.5731507377760632
# Lift (Romance -> Drama): 1.4688926808519822

# El soporte es relativamente alto
# Los géneros Drama y Romance aparecen juntos casi un 7% de las veces.
# La confianza es moderada (un 57% de las veces que aparece Romance, aparece Drama)
# El lift por encima de 1 ofrece una relación positiva. La probabilidad
# de que se de Drama dado Romance es un 47% superior a que si se dieran de manera independiente

# Action, Adventure -> Thriller
# Suppport (Action, Adventure -> Thriller): 0.004350059941770851
# Confidence (Action, Adventure -> Thriller): 0.1814285714285714
# Lift (Action, Adventure -> Thriller): 1.3440261717475621

# El soporte es bajo. Solo un 0.43% de las transacciones contienen Accion, Adventure y Thirller
# La confianza es baja. Solo el 18% de las transacciones con Acción y Aventura
# tienen también Thriller.
# El lift ofrece una asociación débil. La presencia de Action y Adventure aumenta la probabilidad
# de que se de Thriller en un 34%

# Crime, Action -> Thriller
# Suppport (Crime, Action -> Thriller): 0.009624935776674087
# Confidence (Crime, Action -> Thriller): 0.4596510359869138
# Lift (Crime, Action -> Thriller): 3.4051032721740544

# EL soporte es bajo. Un 0.96% de las transacciones contienen Crime, Action y Thriller
# La confianza indica que el 45% de las transacciones con Crime y Action, contienen Thriller.
# El lift es significativamente alto. Es decir, en los casos donde Crime y Action se dan juntos, 
# la probabilidad de que se de Thriller aumenta un 240%.

# Crime -> Action, Thriller
# Suppport (Crime -> Action, Thriller): 0.009624935776674087
# Confidence (Crime -> Action, Thriller): 0.12084288990825687
# Lift (Crime -> Action, Thriller): 3.713692811443747

# El soporte es el mismo que el anterior, bajo.
# La confianza es baja. En un 12% de las transacciones con Crime, aparecen Action y Thriller
# El lift es significativamente alto. La probabilidad de que se den Action y Thriller dado Crime
# es un 270% superior a que si se dieran de manera independiente

# Crime -> Children
# Suppport (Crime -> Children): 0.0007535536907004624
# Confidence (Crime -> Children): 0.009461009174311927
# Lift (Crime -> Children): 0.18332798418851995

# El soporte es bajo. Solo un 0.07% de las transacciones contienen Crime y Children
# La confianza es baja. Solo un 0.94% de las transacciones con Crime contienen Children
# El lift es bajo. La probabilidad de que se de Children dado Crime es un 18% inferior a que si se dieran de manera independiente

####################################################

def _get_support(itemset: list, onehot_dataset: pd.DataFrame) -> float:
    """Calculate the support of an itemset in a onehot dataset

    Parameters
    ----------
    itemset : list
        Itemset
    onehot_dataset : pd.DataFrame
        Onehot dataset with True/False values 
        if the item is present or not

    Returns
    -------
    float
        Support of the itemset
    """
    logic_and_result = onehot_dataset[itemset[0]]
    for item in itemset[1:]:
        logic_and_result = np.logical_and(logic_and_result, onehot_dataset[item])
    return logic_and_result.mean()

def _rule_metrics(antedecent: list, consequent: list, onehot_dataset: pd.DataFrame) -> dict[str, float | str]:
    """Calculate the support, confidence and lift of a rule                                                                                

    Parameters
    ----------
    antedecent : list
        Antedecent of the rule
    consequent : list
        Consequent of the rule
    onehot_dataset : pd.DataFrame
        Onehot dataset with True/False values 
        if the item is present or not

    Returns
    -------
    dict[str, float | str]
        _description_
    """
    itemset = antedecent + consequent
    support = _get_support(itemset=itemset, onehot_dataset=onehot_dataset)
    confidence = support / _get_support(itemset=antedecent, onehot_dataset=onehot_dataset)
    lift = confidence / _get_support(itemset=consequent, onehot_dataset=onehot_dataset)
    
    metrics = {
        'rule': f"{antedecent} -> {consequent}",
        'support': support,
        'confidence': confidence,
        'lift': lift
    }
    return metrics

# Para el cálculo del total de reglas
# de tipo A -> B que se pueden construir para este dataset, se ha 
# construido una función (`get_number_of_rules`) que realiza el cálculo combinatorio
def get_number_of_rules(n_itemset: int, n_subitemset: int) -> int:
    """Calculate the number of rules that can be generated from an itemset and a subitemset

    Parameters
    ----------
    n_itemset : int
        Number of items in the itemset
    n_subitemset : int
        Number of items in the subitemset. For example, if the itemset is [A, B, C]
        and the subitemset is [A, B], n_subitemset = 2

    Returns
    -------
    int
        Number of rules that can be generated
    """
    result = 0
    for n_ant in range(1, n_subitemset):
        result += math.comb(n_itemset, n_subitemset)*math.comb(n_subitemset, n_ant)
    return result

# La formula combinatoria para obtener el número de reglas es usando nCr (Combinaciones de n elementos tomados de r en r).
# Por ejemplo, para el caso de un itemset de 3 elementos y un subitemset de 3 elementos, el número de reglas posibles se deriva de:
# Reglas de tipo A, B -> C (2 elementos en el antecedente y 1 en el consecuente). Su combinación es 3C(3-1), es decir,
# Combinación de 3 elementos tomados de 2 en 2, ya que en el antecedente se toman 2 elementos.
# Reglas de tipo A -> B, C (1 elemento en el antecedente y 2 en el consecuente). Su combinación es 3C(3-2), es decir,
# Combinación de 3 elementos tomados de 1 en 1, ya que en el antecedente se toma 1 elemento.
# En total, se pueden generar la suma de estas dos combinaciones.
# Para el caso de un itemset de 3 elementos y un subitemset de 2 elementos, primero se calcula la combinación de nCr siendo r el subitemset
# y luego se realiza el cálculo de la manera anterior teniendo r como el número de elementos en el antecedente.

def get_rules(itemset: Iterable[str], n_subitemset: int) -> list[tuple[list[str], list[str]]]:
    """Get all the rules from an itemset

    Parameters
    ----------
    itemset : Iterable[str]
        Itemset to generate the rules
    n_subitemset : int
        Number of items in the subitemset. For example, if the itemset is [A, B, C]
        and the subitemset is [A, B], n_subitemset = 2

    Returns
    -------
    list[tuple[list[str], list[str]]]
        List of rules. Each rule is a tuple with the antedecent and the consequent
    """
    rules = []
    for subitemset in list(combinations(itemset, n_subitemset)):
        for i in range(1, len(subitemset)):
            for antecedent in combinations(subitemset, i):
                remaining = set(subitemset) - set(antecedent)
                rules.append((list(antecedent), list(remaining)))
    return rules

def get_rules_metrics_df(rules: list[tuple[list[str], list[str]]], onehot_dataset: pd.DataFrame) -> pd.DataFrame:
    """Get the metrics from a list of rules

    Parameters
    ----------
    rules : list[tuple[list[str], list[str]]]
        List of rules. Each rule is a tuple with the antedecent and the consequent
    onehot_dataset : pd.DataFrame
        Onehot dataset with True/False values 
        if the item is present or not

    Returns
    -------
    pd.DataFrame
        DataFrame with the metrics of the rules and the rule (rule, support, confidence, lift)
    """
    rules_metrics = {
        'rule': [],
        'support': [],
        'confidence': [],
        'lift': []
    }
    for rule in rules:
        rule_metric = _rule_metrics(rule[0], rule[1], onehot_dataset)
        rules_metrics['rule'].append(rule_metric['rule'])
        rules_metrics['support'].append(rule_metric['support'])
        rules_metrics['confidence'].append(rule_metric['confidence'])
        rules_metrics['lift'].append(rule_metric['lift'])
    return pd.DataFrame(rules_metrics)

# # Número de reglas de tipo A -> B para este dataset
# n_subitemset = 2
# print(f"\nNumber of rules of type A -> B: {get_number_of_rules(n_itemset=len(movilens.columns), n_subitemset=n_subitemset)}")

# # Se ha calculado el tiempo que tarda en obtener todas las reglas posibles de tipo A -> B
# start = time.time()
# rules = get_rules(movilens.columns, n_subitemset=2)
# end = time.time()
# rules_metrics_df = get_rules_metrics_df(rules, movilens)
# rules_metrics_df = rules_metrics_df.sort_values(by=['support', 'confidence', 'lift'], ascending=[False, False, False])

# print(f"\nTime elapsed to generate the rules: {end - start} segs. ")

# print(rules_metrics_df.head(20))

# # Número de reglas de tipo A,B -> C ó A -> B,C para este dataset
# n_subitemset = 3
# print(f"\nNumber of rules of type A,B -> C or A -> B,C: {get_number_of_rules(n_itemset=len(movilens.columns), n_subitemset=n_subitemset)}")

# start = time.time()
# rules = get_rules(movilens.columns, n_subitemset=3)
# end = time.time()
# rules_metrics_df = get_rules_metrics_df(rules, movilens)
# rules_metrics_df = rules_metrics_df.sort_values(by=['support', 'confidence', 'lift'], ascending=[False, False, False])

# print(f"\nTime elapsed to generate the rules: {end - start} segs. ")
# print(rules_metrics_df.head(20))

# # Número de reglas de 9 elementos para este dataset. Considero que cada antecendete puede tener un máximo de 8 elementos
# # y un mínimo de 1 elemento
# n_subitemset = 9
# print(f"\nNumber of rules for {n_subitemset} itemset: {get_number_of_rules(n_itemset=len(movilens.columns), n_subitemset=n_subitemset)}")

# start = time.time()
# rules = get_rules(movilens.columns, n_subitemset=9)
# end = time.time()
# print(f"\nTime elapsed to generate the rules: {end - start} segs. ")

# # Número de reglas de todo tipo que contengan desde 1 hasta 19 elementos (todas las reglas posibles)
# n_subitemset = len(movilens.columns)
# print(f"\nNumber of rules for {n_subitemset} itemset: {get_number_of_rules(n_itemset=len(movilens.columns), n_subitemset=n_subitemset)}")

# start = time.time()
# rules = get_rules(movilens.columns, n_subitemset=len(movilens.columns))
# end = time.time()
# print(f"\nTime elapsed to generate the rules: {end - start} segs. ")

# -------------------------------------
rules_2_items = get_rules(movilens.columns, n_subitemset=2)
rules_2_items_metrics_df = get_rules_metrics_df(rules_2_items, movilens)

rules_3_items = get_rules(movilens.columns, n_subitemset=3)
rules_3_items_metrics_df = get_rules_metrics_df(rules_3_items, movilens)

# concat the two dataframes
rules_metrics_df = pd.concat([rules_2_items_metrics_df, rules_3_items_metrics_df])
rules_metrics_df = rules_metrics_df.sort_values(by=['support', 'confidence', 'lift'], ascending=[False, False, False])
print(rules_metrics_df.head(20))

# El soporte mínimo de las reglas es 0.0
print(f"\nMínimo soporte en el dataframe: {rules_metrics_df[(rules_metrics_df['confidence'].notna()) & (rules_metrics_df['lift'].notna())]['support'].min()}")

frequent_itemsets = apriori(movilens, min_support=1e-5, use_colnames=True)
print(frequent_itemsets)

# El dataframe se compone de dos columnas, la primera muestra el support y la segunda los itemsets.
# Se observa, como es evidente, que los itemsets con soporte más bajo son aquellos que contienen más elementos.
# De manera contraria, los itemsets con soporte más alto son aquellos que contienen menos elementos.
# Hay un total de 4076 itemsets con un soporte mínimo de 1e-5

assoc_rules_df = association_rules(frequent_itemsets, num_itemsets=len(movilens))
print(assoc_rules_df)
# OJO: la versión de mlxtend 0.23.3 requiere el argumento num_itemsets para obtener todas las reglas posibles
print(f"\nTotal de reglas generadas: {len(assoc_rules_df)}")

# EL dataframe contiene, a parte de los consecuentes y antecedentes, una serie de métricas
# que sirven parahacer un pruning posterior sobre las reglas a partir de un threshold establecido sobre 
# la metrica deseada.

# Se transforma el dataframe para que muestre "rule", "support", "confidence", "lift"
assoc_rules_df['rule'] = assoc_rules_df.apply(lambda row: f"{list(row['antecedents'])} -> {list(row['consequents'])}", axis=1)
assoc_rules_df = assoc_rules_df[['rule', 'support', 'confidence', 'lift']]
assoc_rules_df = assoc_rules_df.sort_values(by=['support', 'confidence', 'lift'], ascending=[False, False, False])
print(assoc_rules_df.head(20))

print(rules_metrics_df.head(20))

# Curiosamente, las 20 mejores reglas generadas en el punto 2 (reglas de 2 y 3 elementos), son reglas compuestas por 2 items, 1 de antecedente
# Por otro lado, las 20 mejores reglas generadas a partir de la librería mlxtend, son reglas compuestas por más items. Donde el support máximo es
# 0.003168, frente al support de la tabla anterior que es 0.075536. Esto se debe a que la librería mlxtend realiza un pruning de los items
# a partir de `apriori` con la métrica support, de tal manera que itemsets donde la presencia de un item no supere el threshold, no se tienen en cuenta.

## Intersección
# interseccion entre riles_metrics_df y assoc_rules_df
print(f"\nInterseccion entre rules_metrics_df y assoc_rules_df")
rules_intersection = pd.merge(rules_metrics_df, assoc_rules_df, on='rule', how='right', suffixes=('_1', '_2'))
print(rules_intersection)
print(f"\nTotal de reglas en la intersección: {len(rules_intersection)}")

# La intersección entre los dos datarframes da lugar a 11 reglas. Es decir, de las reglas generadas a partir de mlxtend con un theshold de 1e-5
# sobre los items y support con `apriori` y las reglas generadas a partir de la función `get_rules_metrics_df` sin tener en cuenta thresholds, 11 son comunes.
