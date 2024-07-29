import logging
from typing import Callable, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
import matplotlib.pyplot as plt

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Task:
    def __init__(self, name: str, func: Callable, dependencies: List[str] = None):
        self.name = name
        self.func = func
        self.dependencies = dependencies or []
        self.result = None
        self.executed = False

    def execute(self, context: Dict[str, Any]):
        if not self.executed:
            logger.info(f"Executing task: {self.name}")
            try:
                self.result = self.func(context)
                self.executed = True
                logger.info(f"Task {self.name} completed successfully.")
            except Exception as e:
                logger.error(f"Error executing task {self.name}: {e}")
                raise e
        return self.result

class Pipeline:
    def __init__(self):
        self.tasks = {}
        self.execution_order = []
        self.graph = nx.DiGraph()

    def add_task(self, task: Task):
        self.tasks[task.name] = task
        self.graph.add_node(task.name)
        for dep in task.dependencies:
            self.graph.add_edge(dep, task.name)

    def _determine_execution_order(self):
        # Simple topological sort to determine execution order
        visited = set()
        stack = []

        def visit(task_name):
            if task_name in visited:
                return
            visited.add(task_name)
            for dep in self.tasks[task_name].dependencies:
                visit(dep)
            stack.append(task_name)

        for task_name in self.tasks:
            visit(task_name)

        self.execution_order = stack

    def execute(self):
        self._determine_execution_order()
        context = {}
        with ThreadPoolExecutor() as executor:
            future_to_task = {executor.submit(self.tasks[task_name].execute, context): task_name for task_name in self.execution_order}
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result = future.result()
                    context[task_name] = result
                except Exception as exc:
                    logger.error(f"Task {task_name} generated an exception: {exc}")

    def report(self):
        for task_name, task in self.tasks.items():
            status = "Executed" if task.executed else "Pending"
            logger.info(f"Task: {task_name}, Status: {status}, Result: {task.result}")

    def draw_dag(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)
        plt.title("DAG of Pipeline")
        plt.show()
