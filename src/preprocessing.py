
def scale_features(X_train, X_validation, X_test):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_validation_scaled, X_test_scaled

def split_dataset(X, y, validation_size=0.3, test_size=0.3, random_state=42):
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=validation_size + test_size, random_state=random_state)
    validation_size_adjusted = validation_size / (validation_size + test_size)
    X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=validation_size_adjusted, random_state=random_state)

    return X_train, X_validation, X_test, y_train, y_validation, y_test