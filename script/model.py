import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def target_encode(val):
    """
    Encodes the target variable using a mapper dictionary.

    Args:
    val: str, target variable value to be encoded.

    Returns:
    int, encoded target variable value.
    """
    target_mapper = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
    return target_mapper[val]


# Load the penguins dataset
penguins = pd.read_csv("data/penguins.csv")

# Create a copy of the dataset
df = penguins.copy()

# Define the target variable and categorical features to be encoded
target = "species"
encode = ["sex", "island"]

# One-hot encode the categorical features and append them to the dataset
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

# Encode the target variable
df["species"] = df["species"].apply(target_encode)

# Separating X and y
X = df.drop("species", axis=1)
Y = df["species"]

# Build random forest model
clf = RandomForestClassifier()
clf.fit(X, Y)

# Save the model to a file
joblib.dump(clf, "model/penguins_clf.pkl")
