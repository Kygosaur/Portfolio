import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns

class KNNCentroidsPlotter:
    def __init__(self, excel_file, sheet_name):
        self.excel_file = excel_file
        self.sheet_name = sheet_name
        self.df = None
        self.X_train = None
        self.y_train = None
        self.label_encoder = None
        self.knn_model = None
        self.centroids = None
        self.colors = None

    def load_data_from_excel(self):
        self.df = pd.read_excel(self.excel_file, sheet_name=self.sheet_name)

    def apply_pca(self):
        pca = PCA(n_components=2)
        self.X_train = pca.fit_transform(self.X_train)
        self.centroids = pca.transform(self.centroids)

    def preprocess_data(self):
        X = self.df[['ppm', 'Time (second)']].values
        y = self.df['Class']
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.X_train, _, self.y_train, _ = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    def train_model(self, k=3):
        self.knn_model = KNeighborsClassifier(n_neighbors=k)
        self.knn_model.fit(self.X_train, self.y_train)

    def calculate_centroids(self):
        self.centroids = []
        for class_label in np.unique(self.y_train):
            class_centroid = np.mean(self.X_train[self.y_train == class_label], axis=0)
            self.centroids.append(class_centroid)

    def plot_centroids(self):
        x_min, x_max = self.df['ppm'].min() - 1, self.df['ppm'].max() + 1
        time_min, time_max = self.df['Time (second)'].min() - 1, self.df['Time (second)'].max() + 1

        ppm_values = np.linspace(x_min, x_max, 100)
        time_values = np.linspace(time_min, time_max, 100)
        ppm_mesh, time_mesh = np.meshgrid(ppm_values, time_values)
        X_grid = np.column_stack((ppm_mesh.ravel(), time_mesh.ravel()))

        y_grid = self.knn_model.predict(X_grid)

        plt.contourf(ppm_mesh, time_mesh, y_grid.reshape(ppm_mesh.shape), alpha=0.5, cmap='viridis')

        for i, class_label in enumerate(np.unique(self.y_train)):
            plt.scatter(self.centroids[i][0], self.centroids[i][1],
                        marker='o', color=self.colors[i], edgecolor='k', s=100,
                        label=f'Centroid for {self.label_encoder.inverse_transform([class_label])[0]}')

        plt.xlabel('PPM')
        plt.ylabel('Time (second)')
        plt.title('k-NN Centroids')
        plt.legend()
        plt.show()

    def print_centroids(self):
        for i, class_label in enumerate(np.unique(self.y_train)):
            print(f'Centroid for {self.label_encoder.inverse_transform([class_label])[0]}: {self.centroids[i]}')

    def plot_confusion_matrix(self, X_test, y_test):
        y_pred = self.knn_model.predict(X_test)
        target_names = self.label_encoder.inverse_transform(np.unique(self.y_train))
        cm = confusion_matrix(y_test, y_pred)
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, fmt='g', cmap='Blues')  # Plot confusion matrix as heatmap
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(target_names)
        ax.yaxis.set_ticklabels(target_names)
        plt.show()

    def print_classification_report(self, X_test, y_test):
        y_pred = self.knn_model.predict(X_test)
        target_names = self.label_encoder.inverse_transform(np.unique(self.y_train))
        report = classification_report(y_test, y_pred, target_names=target_names)
        print(report)

    def plot_pca_components(self):
        pca = PCA(n_components=2)
        self.X_train = pca.fit_transform(self.X_train)
        explained_variance_ratio = pca.explained_variance_ratio_

        pc1_yield = explained_variance_ratio[0]
        pc2_yield = explained_variance_ratio[1]
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap='viridis')
        plt.xlabel(f'PCA Component 1 (PC Yield: {pc1_yield:.2f})')
        plt.ylabel(f'PCA Component 2 (PC Yield: {pc2_yield:.2f})')
        plt.title('PCA Components')
        plt.colorbar(label='Class')
        plt.show()

    def run(self, k=3, colors=None):
        self.load_data_from_excel()
        self.preprocess_data()
        self.train_model(k)
        self.calculate_centroids()
        self.colors = colors or ['red', 'green', 'blue']
        self.plot_centroids()
        self.print_centroids()
        
        X_test = self.df[['ppm', 'Time (second)']].values
        y_test = self.label_encoder.transform(self.df['Class'])
        self.plot_confusion_matrix(X_test, y_test)
        self.print_classification_report(X_test, y_test)

excel_file = r'c:\Users\Kygo\Py_Projects\Mahina\Sensor Data.xlsx'
sheet_name = 'k-NN MQ2'
knn_plotter = KNNCentroidsPlotter(excel_file, sheet_name)
knn_plotter.run(colors=['red', 'green', 'blue'])
knn_plotter.preprocess_data()
knn_plotter.plot_pca_components()