def plot_confusion_matrix(cm, labels, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plot = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plot

faults = [ff, ef, gf]

xtrain, xtest, ytrain, ytest, xbaltrain, ybaltrain = Turbine.get_test_train_data(features, faults, nf)

labels = ['no-fault', 'feeding fault', 'excitation fault', 'generator fault']

score = 'recall'

tuned_parameters={
        'kernel': ['linear'], 'gamma': ['auto', 1e-3, 1e-4],
        'C': [0.01, .1, 1, 10, 100, 1000],
        'class_weight': [{0: 0.01}, {1: 1}, {1: 2}, {1: 10}, {1: 50}]}


clf = RandomizedSearchCV(SVC(C=1), tuned_parameters, cv=10,
                          scoring='%s_weighted' % score, iid=True)

bgg = BaggingClassifier(base_estimator = clf)

bgg.fit(xbaltrain,ybaltrain)


print("Detailed classification report:")
print()

# Make the predictions
y_true, y_pred = ytest, bgg.predict(xtest)

print(classification_report(y_true, y_pred, target_names=labels))
print()

# Evaluate the SVM using Confusion Matrix
cm = confusion_matrix(ytest, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Also print specificity metric
print("Specificity:", cm[0, 0] / (cm[0, 1] + cm[0, 0]))
print(cm)

# plot the confusion matrices
plot = plot_confusion_matrix(cm_normalized, labels)