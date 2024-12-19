from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Création du pipeline
def build_pipeline(algo_ml, impute_num = SimpleImputer(strategy="median"), impute_var=SimpleImputer(strategy="constant", fill_value="inconnu"), scaler = StandardScaler(),
                   num_vars=None, cat_vars=None):
  # transformer les variables numériques
  if num_vars is None:
      num_vars = []
  if cat_vars is None:
      cat_vars = []
  numeric_transformer = make_pipeline(
      impute_num,
      scaler
    )

  # transformer les variables catégorielles
  categorical_transformer = make_pipeline(
      impute_var,
    (OneHotEncoder(handle_unknown="ignore"))
    )

  # Combinaison des 2 étapes en un seul objet
  preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_vars),
        ('cat', categorical_transformer, cat_vars)
    ]
    )

  # Pipeline final
  model = Pipeline(steps=[('preprocessing', preprocessor),
                        ('regressor', algo_ml)])

  return model