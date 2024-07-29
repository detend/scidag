# SciDAG
A simple library to manage scikit-learn DAGs, allowing for complex workflows with parallel task execution, reporting, and visualization.

## Installation
```bash
pip install scidag
```
## Features
* __Parallel Task Execution:__ Run independent tasks concurrently to speed up your workflow.
* __DAG Visualization:__ Visualize the Directed Acyclic Graph (DAG) of your pipeline.
* __Dynamic Task Execution:__ Execute conditionally based on the results of previous tasks (e.g., selecting the best model).
* __DAG Reporting:__ Generate execution reports to monitor the status of each task.

## Usage
This example demonstrates how to create a pipeline with parallel task execution, DAG visualization, and reporting.

```python
from scidag import Task, Pipeline

def load_data(context):
    from sklearn.datasets import load_iris
    return load_iris(return_X_y=True)

def preprocess_data(context):
    from sklearn.preprocessing import StandardScaler
    X, y = context['load_data']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def train_model_rf(context):
    from sklearn.ensemble import RandomForestClassifier
    X, y = context['preprocess_data']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

def train_model_svm(context):
    from sklearn.svm import SVC
    X, y = context['preprocess_data']
    model = SVC()
    model.fit(X, y)
    return model

def evaluate_model_rf(context):
    from sklearn.metrics import accuracy_score
    X, y = context['preprocess_data']
    model = context['train_model_rf']
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return accuracy

def evaluate_model_svm(context):
    from sklearn.metrics import accuracy_score
    X, y = context['preprocess_data']
    model = context['train_model_svm']
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return accuracy

def select_best_model(context):
    accuracy_rf = context['evaluate_model_rf']
    accuracy_svm = context['evaluate_model_svm']
    if accuracy_rf > accuracy_svm:
        return context['train_model_rf']
    else:
        return context['train_model_svm']

def save_best_model(context):
    best_model = context['select_best_model']
    with open('/path/to/save/best_model.pkl', 'wb') as f:
        import pickle
        pickle.dump(best_model, f)

# Create tasks
load_data_task = Task(name="load_data", func=load_data)
preprocess_data_task = Task(name="preprocess_data", func=preprocess_data, dependencies=["load_data"])
train_model_rf_task = Task(name="train_model_rf", func=train_model_rf, dependencies=["preprocess_data"])
train_model_svm_task = Task(name="train_model_svm", func=train_model_svm, dependencies=["preprocess_data"])
evaluate_model_rf_task = Task(name="evaluate_model_rf", func=evaluate_model_rf, dependencies=["train_model_rf"])
evaluate_model_svm_task = Task(name="evaluate_model_svm", func=evaluate_model_svm, dependencies=["train_model_svm"])
select_best_model_task = Task(name="select_best_model", func=select_best_model, dependencies=["evaluate_model_rf", "evaluate_model_svm"])
save_best_model_task = Task(name="save_best_model", func=save_best_model, dependencies=["select_best_model"])

# Create pipeline and add tasks
pipeline = Pipeline()
pipeline.add_task(load_data_task)
pipeline.add_task(preprocess_data_task)
pipeline.add_task(train_model_rf_task)
pipeline.add_task(train_model_svm_task)
pipeline.add_task(evaluate_model_rf_task)
pipeline.add_task(evaluate_model_svm_task)
pipeline.add_task(select_best_model_task)
pipeline.add_task(save_best_model_task)

# Execute pipeline
pipeline.execute()

# Generate report
pipeline.report()

# Visualize the DAG
pipeline.draw_dag()

```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
