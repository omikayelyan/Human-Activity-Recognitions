from src.data_loader import load_and_prepare_data
from src.models import get_models
from src.train_and_evaluate import train_and_evaluate
from src.augmentation import augment_with_scaled_data


X_train, y_train, X_test, y_test = load_and_prepare_data("data/train.csv", "data/test.csv")

models = get_models()


# ============================================================
# TRAIN AND EVALUATE ORIGINAL DATA
# ============================================================

print("\n==============================")
print("Training on Original Dataset")
print("==============================")

for name, model in models.items():
    trained_model = train_and_evaluate(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        name
    )



# # ============================================================
# # OPTIONAL: TRAIN ON AUGMENTED DATA
# # ============================================================
#
#
# print("\n=====================================")
# print("Training on Augmented Dataset")
# print("=====================================")
#
#
# X_augmented, y_augmented = augment_with_scaled_data(X_train, y_train)
#
#
# for name, model in models.items():
#     trained_model = train_and_evaluate(
#         model,
#         X_augmented,
#         y_augmented,
#         X_test,
#         y_test,
#         name
#     )
