from sklearn.metrics import roc_auc_score
class ModelEvaluator:
    @staticmethod
    def evaluate(model, X_test, y_test):
        y_pred = model.predict(X_test)
        roc_auc = roc_auc_score(y_test, y_pred)
        return {"ROC-AUC": roc_auc}
