import unittest
from scidag import Task, Pipeline

class TestSciDAG(unittest.TestCase):
    def test_simple_pipeline(self):
        def load_data(context):
            return "data"

        def process_data(context):
            data = context['load_data']
            return data + " processed"

        load_task = Task(name="load_data", func=load_data)
        process_task = Task(name="process_data", func=process_data, dependencies=["load_data"])

        pipeline = Pipeline()
        pipeline.add_task(load_task)
        pipeline.add_task(process_task)
        pipeline.execute()

        self.assertEqual(pipeline.tasks["process_data"].result, "data processed")
        pipeline.report()
        pipeline.draw_dag()

if __name__ == "__main__":
    unittest.main()
