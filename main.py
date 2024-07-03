import argparse
import warnings

from llm_src import workflow, ingest_data, run_executors


def play_rag():

    route_executor = run_executors.get_route_executor()
    pandas_executor = run_executors.get_pandas_executor()
    # doc_executor = run_executors.get_doc_executor()
    doc_executor = run_executors.get_doc_executor_history()
    while True:
        user_input = input("Enter something (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        response = workflow.determine_route(
            user_input,
            route_executor=route_executor,
            pandas_executor=pandas_executor,
            doc_executor=doc_executor,
        )
        print(response)
        print("*" * 80)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='play rug', type=str)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    if args.mode == 'load data':
        ingest_data.create_web_data()
    else:
        play_rag()
