# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from matplotlib.colors import ListedColormap

def tensor_operations_example():
    # Example tensor operations with NumPy
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    result = np.dot(a, b)
    print("Tensor Operations Example:")
    print(result)


def turtle_graphics_example():
    # Example turtle graphics
    import turtle

    turtle.forward(100)
    turtle.right(90)
    turtle.forward(100)
    turtle.done()

def gui_example():
   
    from tkinter import Tk, Label, Button

    def on_button_click():
        label.config(text="Button Clicked!")

    root = Tk()
    label = Label(root, text="Hello, GUI!")
    label.pack()
    button = Button(root, text="Click Me", command=on_button_click)
    button.pack()
    root.mainloop()


def data_mining_example():
as
    data = {'Name': ['John', 'Alice', 'Bob'],
            'Age': [25, 30, 22],
            'Salary': [50000, 60000, 55000]}

    df = pd.DataFrame(data)
    print("Data Mining Example DataFrame:")
    print(df)

# Processing example
def processing_example():
 
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

    processed_data = np.mean(data, axis=1)
    print("Processed Data:")
    print(processed_data)

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Train K-NN classifier
    knn_classifier = train_knn_classifier(X_train, y_train)

    # Predicting the Test set results
    y_pred = knn_classifier.predict(X_test)

    # Evaluate the classifier
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)
    print("\nAccuracy Score:", accuracy)
    print("\nClassification Report:")
    print(report)

    # Visualize Training set results
    visualize_results(X_train, y_train, knn_classifier, 'K-NN (Training set)')

    # Visualize Test set results
    visualize_results(X_test, y_test, knn_classifier, 'K-NN (Test set)')

  
    tensor_operations_example()
    turtle_graphics_example()
    gui_example()
    data_mining_example()
    processing_example()

if __name__ == "__main__":
    main()
