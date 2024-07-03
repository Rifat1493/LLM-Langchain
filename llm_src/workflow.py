from llm_src import utils


def determine_route(query, route_executor, pandas_executor, doc_executor):

    route = route_executor.invoke({"question": query})

    if route == "Dataframe":

        try:
            response = pandas_executor.invoke(query)
        except Exception as e:
            response = str(e)
            result = utils.extract_pandas_answer(response)
            return result

    else:
        # return doc_executor.invoke(query)
        result = doc_executor.invoke(
            {"input": query},
            config={"configurable": {"session_id": "abc123"}},
        )["answer"]

        return result

