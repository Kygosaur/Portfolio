import os
import cv2
import glob
import pandas as pd
import numpy as np
import time
import sklearn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from skimage.feature import local_binary_pattern
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import seaborn as sns
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import BaggingClassifier
import joblib

def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cell_size = (4, 4)
    block_size = (2, 2)
    num_orientations = 6
    hog_features = hog(gray_image, orientations=num_orientations,
                       pixels_per_cell=cell_size, cells_per_block=block_size,
                       block_norm='L2-Hys', visualize=False,
                       transform_sqrt=True)
    return hog_features

def extract_lbp_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_radius = 3
    lbp_points = 8 * lbp_radius
    lbp_features = local_binary_pattern(gray_image, lbp_points, lbp_radius).flatten()
    return lbp_features

def extract_hybrid(image):
    hog_features = extract_hog_features(image)
    lbp_features = extract_lbp_features(image)
    hybrid_features = np.concatenate((hog_features.flatten(), lbp_features))
    return hybrid_features

def extract_features_to_dataframe(path, feature_extractor):
    data = []
    file_list = sorted(glob.glob(path))
    for file in file_list:
        try:
            image = cv2.imread(file)
            if image is None:
                print(f"Failed to read image: {file}")
                continue
            features = feature_extractor(image)
            class_label = os.path.basename(file).split(" ")[0]
            data.append([file, features.flatten(), class_label])
        except Exception as e:
            print(f"Error processing image: {file}")
            print(f"Error message: {str(e)}")
    df = pd.DataFrame(data, columns=['File Path', 'Features', 'Class Label'])

    # Add Classification Report (Overall) to the DataFrame
    X = df['Features'].tolist()
    y = df['Class Label'].tolist()
    svm_model = SVC()
    svm_model.fit(X, y)
    y_pred = svm_model.predict(X)
    classification_report_all = classification_report(y, y_pred, output_dict=True)
    classification_report_df = pd.DataFrame(classification_report_all).transpose()
    df['Classification Report'] = classification_report_df['support']
    return df, classification_report_df

def plot_confusion_matrix(y, y_pred, class_names):
    conf_matrix = confusion_matrix(y, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='g', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()

def plot_roc_curves(y_true, y_scores, classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, cls in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_scores[:, i], pos_label=cls)
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    for i, cls in enumerate(classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {cls}, area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.show()

def plot_multi_class_roc_curves(y_true, y_scores, classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, cls in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == cls, y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    for i, cls in enumerate(classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {cls}, area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.show()

def run_feature_extraction_and_svm(input_path, feature_extractor, model_type, num_folds=4):
    start_time = time.time()
    df, classification_report_df = extract_features_to_dataframe(input_path, feature_extractor)
    X_hog = df['Features'].tolist()
    X_lbp = df['Features'].tolist()
    y = df['Class Label'].tolist()
    feature_selector_hog = SelectKBest(score_func=f_classif, k=20)
    X_hog_selected = feature_selector_hog.fit_transform(X_hog, y)
    feature_selector_lbp = SelectKBest(score_func=f_classif, k=20)
    X_lbp_selected = feature_selector_lbp.fit_transform(X_lbp, y)
    X_selected = np.concatenate((X_hog_selected, X_lbp_selected), axis=1)
    estimator = SVC(kernel='rbf', C=0.1, decision_function_shape='ovr')
    svm_model_bagging = BaggingClassifier(estimator, n_estimators=10)
    # Perform k-fold cross-validation
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    scores = []
    y_pred_total = []
    y_score_total = []
    for train_index, test_index in skf.split(X_selected, y):
        X_train, X_test = np.array(X_selected)[train_index], np.array(X_selected)[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        svm_model_bagging.fit(X_train, y_train)
        model_filename = f"{model_type}_model.joblib"
        joblib.dump(svm_model_bagging, model_filename)
        y_pred = svm_model_bagging.predict(X_test)
        df.loc[test_index, 'Predicted Label'] = y_pred  # Add Predicted Label column to DF
        y_pred_total.extend(y_pred)
        # Calculate predicted probabilities
        y_score = svm_model_bagging.decision_function(X_test)
        y_score_total.extend(y_score)
        # Compute precision, recall, f1-score, and support
        classification_report_fold = classification_report(y_test, y_pred, output_dict=True)
        scores.append(classification_report_fold['weighted avg'])
    classification_report_all = sklearn.metrics.classification_report(y, y_pred_total, output_dict=True)
    classification_report_df = pd.DataFrame(classification_report_all).transpose()
    end_time = time.time()
    duration = end_time - start_time
    print(f"Runtime Duration (SVM): {duration:.4f} seconds")
    # Compute average scores across folds
    scores_df = pd.DataFrame(scores)
    scores_avg = scores_df.mean().to_frame().transpose()
    scores_avg.index = ['avg/total']
    scores_df = scores_df.append(scores_avg)
    print("Average Scores (Cross-Validation):\n")
    print(scores_df)
    plot_multi_class_roc_curves(np.array(y), np.array(y_score_total), svm_model_bagging.classes_)
    plot_confusion_matrix(y_test, y_pred, df['Class Label'].unique())
    # Save Classification Report (Overall) and per-model report to Excel file
    excel_filename = 'FYP.xlsx'
    if os.path.exists(excel_filename):
        os.remove(excel_filename)
    with pd.ExcelWriter(excel_filename) as writer:
        classification_report_df.to_excel(writer, sheet_name='Classification Report (Overall)', index=False)
        scores_df.to_excel(writer, sheet_name='Average Scores (CV)', index=False)
    wb = writer.book
    if model_type == "hog":
        sheet_name = 'hog'
    elif model_type == "lbp":
        sheet_name = 'lbp'
    elif model_type == "hybrid":
        sheet_name = 'hybrid'
    df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2, startcol=0)
    wb[sheet_name].title = sheet_name
    writer.save()

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(f"Average Scores (CV) saved to 'Average Scores (CV)' sheet")
    print(f"Classification Report ({model_type}) saved to '{sheet_name}' sheet")

training_path = r'FYP\Images\Training\*.jpg'
print(f"Class Input: {len(set(os.path.basename(file).split(' ')[0] for file in glob.glob(training_path)))}")
print(f"Input Images from Path: {len(glob.glob(training_path))}")

files = sorted(glob.glob(training_path))
labels = [os.path.basename(file).split(" ")[0] for file in files]
df, classification_report_df = extract_features_to_dataframe(training_path, extract_lbp_features)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Features'].tolist(), df['Class Label'].tolist(), test_size=0.4, random_state=42)

# Run feature extraction and SVM with LBP features
run_feature_extraction_and_svm(training_path, extract_lbp_features, "lbp")
#lbp_model = joblib.load("lbp_model.joblib")
# Run feature extraction and SVM with HOG features
#run_feature_extraction_and_svm(training_path, extract_hog_features, "hog")
#hog_model = joblib.load("hog_model.joblib")
# Run feature extraction and SVM with hybrid features
#run_feature_extraction_and_svm(training_path, extract_hybrid, "hybrid")
#hybrid_model = joblib.load("hybrid_model.joblib")
