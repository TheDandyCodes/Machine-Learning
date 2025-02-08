from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import bernoulli, multinomial, norm
from tqdm import tqdm


class NaiveBayes:
    """Naive Bayes classifier."""

    def __init__(self):
        self.parameters = {}
        self.probabilities = {}
        self.classes = []

    def _set_classes(self, y: pd.Series):
        """Set the classes of the model.

        Parameters
        ----------
        y : pd.Series
            Target variable.
        """
        self.classes = np.unique(y)

    def _gauss_likelihood(
        self, feature: pd.Series, mean: float, std: float
    ) -> pd.Series:
        """Calculate the likelihood of a feature given a mean and a standard deviation.

        Parameters
        ----------
        feature : pd.Series
            Feature to calculate the likelihood.
        mean : float
            Mean of the values of the feature per class.
        std : float
            Standard deviation of the values of the feature per class.

        Returns
        -------
        pd.Series
            Likelihood of the feature given the mean and the standard deviation.
        """
        exponent = np.exp(-1 / 2 * ((feature - mean) / std) ** 2)
        function = 1 / (std * np.sqrt(2 * np.pi)) * exponent
        # Impose a lower bound on the probabilities to avoid extremeley small
        # values that can lead to values of zero, therefore indefinite values
        # when calculating the log.
        return np.maximum(function, 1e-10)

    def _discret_likelihood_with_laplace_smth(
        self, X_train: pd.DataFrame, y_train: pd.Series, column: str, alpha: int = 1
    ) -> dict[str, dict[str, float]]:
        """Calculate likelihood of discrets values of a feature using Laplace smoothing.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data.
        y_train : pd.Series
            Training target.
        column : str
            Column to calculate the likelihood.
        alpha : int, optional
            Smoothing parameter. Defaults to 1.

        Returns
        -------
        pd.Series
            Likelihood of each value for the feature.
        """
        dataset = pd.concat([X_train, y_train], axis=1)
        result = (
            dataset.groupby(dataset.columns[-1])[column]
            .value_counts()
            .unstack(fill_value=0)
        )
        result = result.reindex(columns=dataset[column].unique(), fill_value=0)

        # `.values[:, None]` converts Unidimensional array
        # to bidimensional array. From (2,) to (2, 1)
        laplace_smth_result = (result + alpha) / (
            result.sum(axis=1) + alpha * len(X_train[column].unique())
        ).values[:, None]

        laplace_smth_result_dict = laplace_smth_result.to_dict(orient="index")
        return laplace_smth_result_dict

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Calculate the mean and the standard deviation of the features per class.

        As well as the discrete likelihood of the cathegories
        that belong to discreat features.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data.
        y_train : pd.Series
            Target of training data.
        """
        self._set_classes(y_train)
        parameters = {}
        for class_ in self.classes:
            parameters[class_] = {
                "apriori": len(y_train[y_train == class_]) / len(y_train)
            }
            for feature in X_train.select_dtypes(include="number").columns:
                parameters[class_][feature] = {}
                parameters[class_][feature]["mean"] = X_train[y_train == class_][
                    feature
                ].mean()
                parameters[class_][feature]["std"] = X_train[y_train == class_][
                    feature
                ].std()

            for feature in X_train.select_dtypes(include=["object", "bool"]).columns:
                laplace_smth_result_dict = self._discret_likelihood_with_laplace_smth(
                    X_train, y_train, feature
                )
                parameters[class_][feature] = laplace_smth_result_dict[class_]

        self.parameters = parameters

    def predict_prob(self, X_test: pd.DataFrame) -> dict[str, pd.Series]:
        """Predicts the probability of each class given the features.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test data.

        Returns
        -------
        dict[str, pd.Series]
            Probabilities per class.
        """
        probabilities = {}
        for class_ in self.classes:
            likelihood = 1
            for feat in X_test.columns:
                if X_test[feat].dtype == "bool":  # Binary variable (Bernoulli)
                    # The Bernoulli PMF also uses a ratio (p) that is equivalent to
                    # the relative frequency in the dataset.
                    likelihood *= bernoulli.pmf(
                        k=X_test[feat],
                        p=self.parameters[class_][feat][True],
                    )  # parameter p is the probability of the class being 'yes'.
                    # If the class is 'no', the probability is 1 - p
                elif X_test[feat].dtype == "object":
                    # Categorical variable using Multinomial distribution
                    likelihood *= multinomial.pmf(
                        x=np.array(pd.get_dummies(X_test[feat])),
                        # Since only one category per row is observed
                        # (e.g., a single category of marital), n is set to 1.
                        n=1,
                        p=[
                            self.parameters[class_][feat][col]
                            for col in pd.get_dummies(X_test[feat]).columns
                        ],  # Same order as the columns of the dummies
                    )
                else:  # Continuous variable using Gaussian distribution
                    likelihood *= norm.pdf(
                        X_test[feat],
                        self.parameters[class_][feat]["mean"],
                        self.parameters[class_][feat]["std"],
                    )
            probabilities[class_] = likelihood * self.parameters[class_]["apriori"]

        return probabilities

    def predict_log_prob(self, X_test: pd.DataFrame) -> dict[str, pd.Series]:
        """Predicts the log probability of each class given the features.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test data.

        Returns
        -------
        dict[str, pd.Series]
            Log-probabilities per class.
        """
        log_probabilities = {}
        for class_ in self.classes:
            likelihood = 0
            for feat in X_test.columns:
                if X_test[feat].dtypes == "bool":  # Binary variable (Bernoulli)
                    # The Bernoulli PMF also uses a ratio (p) that is equivalent to
                    # the relative frequency in the dataset.
                    likelihood += bernoulli.logpmf(
                        k=X_test[feat],
                        p=self.parameters[class_][feat][True],
                    )  # parameter p is the probability of the class being 'yes'.
                    # If the class is 'no', the probability is 1 - p
                elif X_test[feat].dtype == "object":  # Cat variable using Multinomial
                    likelihood += multinomial.logpmf(
                        x=np.array(pd.get_dummies(X_test[feat])),
                        # Since only one category per row is observed
                        # (e.g., a single category of marital), n is set to 1.
                        n=1,
                        p=[
                            self.parameters[class_][feat][col]
                            for col in pd.get_dummies(X_test[feat]).columns
                        ],  # Same order as the columns of the dummies
                    )
                else:  # Continuous variable using Gaussian distribution
                    likelihood += norm.logpdf(
                        X_test[feat],
                        self.parameters[class_][feat]["mean"],
                        self.parameters[class_][feat]["std"],
                    )
            log_probabilities[class_] = likelihood + np.log(
                self.parameters[class_]["apriori"]
            )
        return log_probabilities

    def predict(self, X_test: pd.DataFrame, method: Optional[str] = None) -> pd.Series:
        """Predicts the class of the features.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test data.
        method : None | str, optional
            Method to predict the class. This can be None or 'log'.
            Defaults to None.

        Returns
        -------
        pd.Series
            Predicted class.
        """
        if method == "log":  # noqa: SIM108
            probabilities = pd.DataFrame(self.predict_log_prob(X_test))
        else:
            probabilities = pd.DataFrame(self.predict_prob(X_test))

        return probabilities.idxmax(axis="columns")

    def _cross_validation_split(
        self, k: int, data: pd.DataFrame
    ) -> list[dict[str, pd.DataFrame]]:
        """Split the dataset into k folds.

        Parameters
        ----------
        k : int
            Number of folds.
        data : pd.DataFrame
            Dataset.

        Returns
        -------
        list[dict[str, pd.DataFrame]]
            list of dictionaries with the train and test sets.
        """
        data = data.sample(frac=1).reset_index(drop=True)
        data["fold"] = data.index % k
        folds = []
        for i in range(k):
            test = data[data["fold"] == i].drop(columns="fold")
            train = data[data["fold"] != i].drop(columns="fold")
            folds.append({"train": train, "test": test})
        return folds

    def accuracy_metric(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate the accuracy of a prediction.

        Parameters
        ----------
        y_true : pd.Series
            True target.
        y_pred : pd.Series
            Predicted target.

        Returns
        -------
        float
            Accuracy of a prediction.
        """
        return (y_true == y_pred).sum() / len(y_true)

    def cross_validation_evaluate(
        self, k: int, data: pd.DataFrame, method: Optional[str] = None
    ) -> tuple[list[float], float]:
        """Evaluate the model using cross-validation.

        Parameters
        ----------
        k : int
            Number of folds.
        data : pd.DataFrame
            Dataset.
        method : None | str, optional
            Method to predict the class. This can be None or 'log'.
            Defaults to None.

        Returns
        -------
        tuple[list[float], float]
            Tuple with list of accuracies and mean accuracy.
        """
        folds = self._cross_validation_split(k=k, data=data)
        accuracies = []
        for fold in tqdm(folds, desc="Folds", total=k):
            # Reset index to avoid problems with the index of the data
            # when extracting probabilities using scipy distributions.
            X_train = fold["train"].iloc[:, :-1].reset_index(drop=True)
            y_train = fold["train"].iloc[:, -1].reset_index(drop=True)
            X_test = fold["test"].iloc[:, :-1].reset_index(drop=True)
            y_test = fold["test"].iloc[:, -1].reset_index(drop=True)
            self.fit(X_train=X_train, y_train=y_train)
            y_pred = self.predict(X_test=X_test, method=method)

            accuracy = self.accuracy_metric(y_true=y_test, y_pred=y_pred)
            accuracies.append(accuracy)
        mean_accuracy = np.mean(accuracies)
        return accuracies, mean_accuracy


if __name__ == "__main__":
    print("Loading datasets...\n\n")
    from ucimlrepo import fetch_ucirepo

    def two_categories_to_binary(df: pd.DataFrame) -> pd.DataFrame:
        """Encode two categories variables into binary variables.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with some columns with two categories

        Returns
        -------
        pd.DataFrame
            Dataframe with encoded variables
        """
        # Select columns with only two categories
        binary_columns = (
            df.select_dtypes(include="object")
            .nunique()[df.select_dtypes(include="object").nunique() == 2]
            .index
        )
        for col in binary_columns:
            df[col] = pd.get_dummies(df[col], drop_first=True)
        return df

    iris_dataset = fetch_ucirepo(id=53)
    X_iris = iris_dataset.data.features
    y_iris = iris_dataset.data.targets["class"]
    iris_df = pd.concat([X_iris, y_iris], axis=1)

    bank_marketing = fetch_ucirepo(id=222)
    # Secure independent copy to avoid view problems
    X_bank_marketing = bank_marketing.data.features.copy()
    # Replace 'yes' and 'no' with binary values
    for col in ["default", "housing", "loan"]:
        X_bank_marketing[col] = X_bank_marketing[col].map({"yes": True, "no": False})
        # Para las variables con 2 categorías (no binarias), se introducirán en la distribución Multinomial,
        # ya que en casos como la variable `stalk-shape`, sus valores `enlarging=e`,`tapering=t` no son conceptualmente opuestos,
        # como una varibale binaria. Es decir, si asumieramos que elarging es True y tapering es False,
        # estaríamos concluyendo que concepturalmente elarging es lo opuesto a tapering.

        # De esta manera, si codificaramos la variable `stalk-shape` como binaria, tendriamos que escoger
        # qué categoría considerar como True y cual como False, haciendo que realmente la categoría que cuenta,
        # conceptualmente, sea a la que le asociamos el valor de True, por ejemplo elarging. De esta manera la
        # variable sería menos interpretable. Es decir, sería necesario que esa variable se llamara Elarging y
        # que sus valores fueran True o False, es decir, está ensanchando (True) o no está ensanchando (False).
        # Por eso mismo, solo se utilizará la distribución Bernouilli para las variables conceptualmente binarias,
        # como puede ser `default`, donde True equivaldría a que se ha dado default y False que no.
    y_bank_marketing = bank_marketing.data.targets["y"]
    bank_marketing_df = pd.concat([X_bank_marketing, y_bank_marketing], axis=1)
    bank_marketing_df_no_misssing = bank_marketing_df.dropna(
        subset=["job", "education"]
    )
    bank_marketing_df_no_misssing = bank_marketing_df_no_misssing.drop(
        columns=["contact", "poutcome"]
    )

    mushrooms = fetch_ucirepo(id=73)
    X_mushrooms = mushrooms.data.features.copy()
    X_mushrooms = X_mushrooms.drop(
        columns="stalk-root"
    )  # Drop column w/ missing values
    # Replace 'f' and 't' with binary values
    X_mushrooms["bruises"] = X_mushrooms["bruises"].map({"f": False, "t": True})
    y_mushrooms = mushrooms.data.targets  # p = poisonous, e = edible
    mushrooms_df = pd.concat([X_mushrooms, y_mushrooms], axis=1)

    print("## IRIS DATASET ##\n")
    print("Fitting the models...\n")
    nb = NaiveBayes()

    print("Evaluating the model with NO log-probabilities...\n")
    cv_ev = nb.cross_validation_evaluate(k=5, data=iris_df)
    for i, fold_acc in enumerate(cv_ev[0]):
        print(f"Fold {i} - Accuracy: {fold_acc}")
    print(f"Mean Accuracy: {cv_ev[1]}\n\n")

    print("Evaluating the model with log-probabilities...\n")
    cv_ev_log = nb.cross_validation_evaluate(k=5, data=iris_df, method="log")
    for i, fold_acc in enumerate(cv_ev_log[0]):
        print(f"Fold {i} - Accuracy: {fold_acc}")
    print(f"Mean Accuracy: {cv_ev_log[1]}\n\n")

    print("## BANK MARKETING DATASET ##\n")
    print("Fitting the models...\n")
    nb = NaiveBayes()

    print("Evaluating the model with NO log-probabilities...\n")
    cv_ev = nb.cross_validation_evaluate(k=5, data=bank_marketing_df_no_misssing)
    for i, fold_acc in enumerate(cv_ev[0]):
        print(f"Fold {i} - Accuracy: {fold_acc}")
    print(f"Mean Accuracy: {cv_ev[1]}\n\n")

    print("Evaluating the model with log-probabilities...\n")
    cv_ev_log = nb.cross_validation_evaluate(
        k=5, data=bank_marketing_df_no_misssing, method="log"
    )
    for i, fold_acc in enumerate(cv_ev_log[0]):
        print(f"Fold {i} - Accuracy: {fold_acc}")
    print(f"Mean Accuracy: {cv_ev_log[1]}\n\n")

    print("## MUSHROOMS DATASET ##\n")
    print("Fitting the models...\n")
    nb = NaiveBayes()

    print("Evaluating the model with NO log-probabilities...\n")
    cv_ev = nb.cross_validation_evaluate(k=5, data=mushrooms_df)
    for i, fold_acc in enumerate(cv_ev[0]):
        print(f"Fold {i} - Accuracy: {fold_acc}")
    print(f"Mean Accuracy: {cv_ev[1]}\n\n")

    print("Evaluating the model with log-probabilities...\n")
    cv_ev_log = nb.cross_validation_evaluate(k=5, data=mushrooms_df, method="log")
    for i, fold_acc in enumerate(cv_ev_log[0]):
        print(f"Fold {i} - Accuracy: {fold_acc}")
    print(f"Mean Accuracy: {cv_ev_log[1]}\n\n")

    # En esta modificaciíon del código se emplean las distribuciones
    # Multinomial, Bernoulli y Normal para calcular las probabilidades.
    # Esto no cambia el valor de las probabilidades de una feature por clase.
    # Por un lado, la distribución Bernoulli usa el ratio (p), que es equivalente a
    # la frecuencia relativa en el dataset, es decir, para una variable binaria,
    # la frecuencia relativa del valor de la variable para cada clase es equivalente
    # a la probabilidad de la variable para cada clase según la distribución.
    # Por otro lado, la distribución Multinomial se emplea para variables categóricas
    #

    # ## Resultados de la evalaución del modelo con el dataset IRIS ##
    # Evaluating the model with NO log-probabilities...

    # Fold 0 - Accuracy: 1.0
    # Fold 1 - Accuracy: 0.9
    # Fold 2 - Accuracy: 0.9333333333333333
    # Fold 3 - Accuracy: 1.0
    # Fold 4 - Accuracy: 0.9333333333333333
    # Mean Accuracy: 0.9533333333333334

    # Estos resultados indican que el modelo tiene un buen desempeño en términos
    # de Accuracy (Predicciones correctas / Total de predicciones)
    # en la clasificación de las flores del dataset Iris (Setosa, Versicolor, Virginica)
    # Esto quiere decir que el dataset contiene información suficiente para que
    # el modelo pueda aprender y generalizar correctamente
    # a partir del cálculo de las probabilidades condicionales de las features.

    # Evaluating the model with log-probabilities...

    # Fold 0 - Accuracy: 1.0
    # Fold 1 - Accuracy: 0.9666666666666667
    # Fold 2 - Accuracy: 0.9
    # Fold 3 - Accuracy: 0.9
    # Fold 4 - Accuracy: 0.9666666666666667
    # Mean Accuracy: 0.9466666666666667

    # En este caso, se emplea el logaritmo de las probabilidades para evitar problemas
    # de underflow, dado que las probabilidades son valores muy pequeños pudiendo ser
    # cero en algunos casos. El desempeño es similar al modelo sin logaritmos

    # ## Resultados de la evalaución del modelo con el dataset BANK MARKETING ##
    # Evaluating the model with NO log-probabilities...

    # Fold 0 - Accuracy: 0.8837828452367172
    # Fold 1 - Accuracy: 0.8745225141798819
    # Fold 2 - Accuracy: 0.8732492186595671
    # Fold 3 - Accuracy: 0.8765918036582542
    # Fold 4 - Accuracy: 0.8831905533688353
    # Mean Accuracy: 0.878267387020651

    # En este caso, el modelo tiene un desempeño aceptable en términos de Accuracy
    # en la clasificación de los clientes (Sí se suscribe a un depósito, No se suscribe
    # a un depósito). Dado que el dataset contiene más complejidad en términos de la
    # cantidad de features y la cantidad de datos, el modelo tiene un desempeño menos
    # preciso que en el caso del dataset Iris.

    # Evaluating the model with log-probabilities...

    # Fold 0 - Accuracy: 0.8747540224563028
    # Fold 1 - Accuracy: 0.8764903345294595
    # Fold 2 - Accuracy: 0.8818150248871397
    # Fold 3 - Accuracy: 0.8768233387358185
    # Fold 4 - Accuracy: 0.880064829821718
    # Mean Accuracy: 0.8779895100860877

    # En este caso, se emplea el logaritmo de las probabilidades para evitar problemas
    # de underflow. El desempeño es similar al modelo sin logaritmos.

    # ## Resultados de la evalaución del modelo con el dataset MUSHROOMS ##
    # Evaluating the model with NO log-probabilities...

    # Fold 0 - Accuracy: 0.9501538461538461
    # Fold 1 - Accuracy: 0.9661538461538461
    # Fold 2 - Accuracy: 0.963076923076923
    # Fold 3 - Accuracy: 0.96
    # Fold 4 - Accuracy: 0.9636699507389163
    # Mean Accuracy: 0.9606109132247063

    # Evaluating the model with log-probabilities...

    # Fold 0 - Accuracy: 0.9581538461538461
    # Fold 1 - Accuracy: 0.9643076923076923
    # Fold 2 - Accuracy: 0.9655384615384616
    # Fold 3 - Accuracy: 0.9575384615384616
    # Fold 4 - Accuracy: 0.9593596059113301
    # Mean Accuracy: 0.9609796134899582
