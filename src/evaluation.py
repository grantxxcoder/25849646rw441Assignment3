def calculate_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def calculate_precision(y_true, y_pred, average='weighted'):
    from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred, average=average)

def calculate_recall(y_true, y_pred, average='weighted'):
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, average=average)

def calculate_f1_score(y_true, y_pred, average='weighted'):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average=average)

def generate_classification_report(y_true, y_pred, target_names):
    from sklearn.metrics import classification_report
    return classification_report(y_true, y_pred, target_names=target_names)